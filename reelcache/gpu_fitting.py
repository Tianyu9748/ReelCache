"""Explicit owner-thread GPU fitting backends with matched stored-basis projection."""

import contextlib
import os
import threading
import time

import torch

from .errors import CapacityError, ConfigurationError, StateError, UnsupportedError
from .exact_blas import mm
from .factorization import (
    Factors, _canonical_signs, fast_condition_bound, fit_basis, project_onto_basis,
    validate_matrix, validate_rank,
)


GPU_FIT_METHODS = ("gesvd", "gesvdj", "gesvda", "torch_lowrank", "cupy", "randomized", "gram", "gram_batched")
EIGH_SOLVERS = ("auto", "torch_batched", "cusolver_ctypes", "torch_loop")
# Torch versions from which one CUDA ``torch.linalg.eigh`` on a stacked [b, n, n] batch
# (n > 32) dispatches to ``cusolverDnXsyevBatched`` (measured: Torch 2.13; Torch 2.8
# loops ``syevd`` per matrix for n > 32 and batches only n <= 32 through syevjBatched).
TORCH_BATCHED_XSYEV_MIN_VERSION = (2, 13)


def _torch_version() -> tuple[int, int]:
    parts = torch.__version__.split("+")[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise UnsupportedError(f"cannot parse the running Torch version {torch.__version__!r}") from exc


def torch_batched_eigh_is_xsyevbatched() -> bool:
    """True when this Torch build sends a batched CUDA eigh (n > 32) to cusolverDnXsyevBatched."""
    return torch.version.cuda is not None and _torch_version() >= TORCH_BATCHED_XSYEV_MIN_VERSION


def _torch_rows_eigh(grams):
    values, columns = torch.linalg.eigh(grams)
    return values, columns.transpose(1, 2)


def _torch_loop_eigh(grams):
    pairs = [torch.linalg.eigh(gram) for gram in grams]
    return torch.stack([values for values, _ in pairs]), torch.stack([columns.T for _, columns in pairs])


#: Share of the device's total memory one batched eigensolver call may use (input copy, vectors, workspace).
#: A device constant, so the sub-batch size is deterministic per GPU model and batch shape.
EIGH_MEMORY_FRACTION = 0.125
#: Without the ctypes buffer-size query, batches above this many Gram elements are split (cuSOLVER rejects
#: 60 x 4096^2 = 1.0e9 in cusolverDnXsyevBatched_bufferSize with status 3; 32 x 4096^2 and 60 x 3072^2 work).
EIGH_UNQUERIED_ELEMENT_LIMIT = 1 << 29
_CUSOLVER_INVALID_VALUE = 3
#: From this matrix size on, :class:`BoundedEigh` solves every matrix in its own call by default.
#: ``cusolverDnXsyevBatched`` returns eigenpairs of a matrix that depend on the batch size of its call -- sizes fall
#: into classes such as {1} {2} {3..8} {9} {10, 11} {12..30} {31..33} for n >= 1024 (RTX PRO 6000 and L40S, Torch 2.8
#: ctypes and Torch 2.13 / CUDA 13) -- but never on the content or position of the other matrices of the call. One
#: matrix per call therefore makes its eigenpairs a function of the matrix alone (independent of the chunk's fit set
#: and of a memory-bounded planner's groups), and from 4096 on it costs nothing: RTX PRO 6000 4096^2 69.5-72.9 ms
#: alone vs 65-74 ms per matrix in batches of 12-32, 5120^2 119 ms vs 144-150 ms; L40S 4096^2 82 vs 113-137 ms.
PER_MATRIX_EIGH_MIN_N = 4096


def default_eigh_sub_batch(n: int) -> int | None:
    """Default fixed sub-batch of a batched eigensolve of ``n x n`` matrices: 1 from PER_MATRIX_EIGH_MIN_N on, else
    None (the whole batch in one call, split only when cuSOLVER rejects its size or the memory bound requires)."""
    return 1 if n >= PER_MATRIX_EIGH_MIN_N else None


def xsyevbatched_workspace_bytes(batch: int, n: int, device) -> int | None:
    """Device workspace bytes of one ``cusolverDnXsyevBatched`` call, or None when cuSOLVER rejects the size.

    Uses the ctypes binding's handle on ``device``; raises UnsupportedError when the binding is unavailable
    or the query fails for another reason.
    """
    import ctypes
    from . import cusolver_eigh
    if not cusolver_eigh.available():
        raise UnsupportedError("the cusolverDnXsyevBatched buffer-size query needs the ctypes binding")
    device = torch.device(device)
    lib, handle, params = cusolver_eigh._handle(device)
    probe = torch.empty(1, dtype=torch.float32, device=device)
    device_bytes, host_bytes = ctypes.c_size_t(), ctypes.c_size_t()
    with torch.cuda.device(device):
        status = lib.cusolverDnXsyevBatched_bufferSize(
            handle, params, cusolver_eigh._EIG_MODE_VECTOR, cusolver_eigh._FILL_LOWER, ctypes.c_int64(n),
            cusolver_eigh._CUDA_R_32F, ctypes.c_void_p(probe.data_ptr()), ctypes.c_int64(n),
            cusolver_eigh._CUDA_R_32F, ctypes.c_void_p(probe.data_ptr()), cusolver_eigh._CUDA_R_32F,
            ctypes.byref(device_bytes), ctypes.byref(host_bytes), ctypes.c_int64(batch),
        )
    if status == _CUSOLVER_INVALID_VALUE:
        return None
    if status != 0:
        raise UnsupportedError(f"cusolverDnXsyevBatched_bufferSize failed with status {status}")
    return int(device_bytes.value)


class BoundedEigh:
    """A batched eigensolver called on sub-batches that cuSOLVER accepts and the device memory bound admits.

    The solver never changes. ``sub_batch_size(n)`` (default :func:`default_eigh_sub_batch`) fixes the sub-batch of
    ``n x n`` problems: a fixed size ``s`` solves every call in ``ceil(b / s)`` calls of exactly ``s`` matrices (the
    last padded with identity matrices whose results are dropped). Since ``cusolverDnXsyevBatched``'s eigenpairs
    of a matrix depend on its call's batch size but not on the other matrices, a fixed size makes each matrix's
    eigenpairs independent of what it is solved with (bitwise; e.g. by a memory-bounded planner group); the
    default is one matrix per call from PER_MATRIX_EIGH_MIN_N (4096) on, where it costs nothing. A fixed size is
    never changed (that would change the eigenpairs): cuSOLVER rejecting it raises UnsupportedError, and a fixed
    sub-batch whose copies and workspace exceed ``memory_fraction`` of the device memory (or ``total_memory``) runs
    as the minimum unit, recorded ``over_budget`` (like one matrix of the halving path). Without a fixed size
    (default below 4096) a batch is solved in one call, or -- when cuSOLVER
    rejects its size (buffer-size status 3, e.g. LongCat's 60 Grams at n = 4096) or the bound does not admit it --
    in equal consecutive sub-batches of the largest admissible size (halving from the whole batch); there a
    matrix's eigenpairs depend on the size class of its (sub-)batch. Convergence flags of a solver exposing
    ``unchecked`` are read once per call. ``calls`` records every call's ``batch``, ``n``, ``sub_batch``,
    ``solver_calls``, ``reason`` (and ``padded`` when padded); ``last`` the latest.
    """

    def __init__(self, label: str, solver, *, memory_fraction: float = EIGH_MEMORY_FRACTION,
                 workspace=None, total_memory: int | None = None, sub_batch_size=None):
        if not 0 < memory_fraction <= 1:
            raise ConfigurationError("eigh memory fraction must be in (0, 1]")
        if total_memory is not None and (type(total_memory) is not int or total_memory <= 0):
            raise ConfigurationError("total_memory must be a positive byte count")
        if sub_batch_size is not None and not callable(sub_batch_size):
            raise ConfigurationError("sub_batch_size must map n to a fixed sub-batch size or None")
        self.label, self.solver, self.memory_fraction = label, solver, memory_fraction
        # ``workspace(batch, n, device)`` -> bytes or None (rejected) and ``total_memory`` replace the device
        # queries (tests, explicit budgets); with ``total_memory`` the bound applies on any device.
        self._workspace = workspace
        self.total_memory = total_memory
        self.sub_batch_size = default_eigh_sub_batch if sub_batch_size is None else sub_batch_size
        self.calls: list[dict] = []
        self.last: dict | None = None

    def derive(self, *, memory_fraction: float | None = None, total_memory: int | None = None) -> "BoundedEigh":
        """The same solver and sub-batch policy under another memory bound (e.g. a planner's working-set budget)."""
        return BoundedEigh(self.label, self.solver,
                           memory_fraction=self.memory_fraction if memory_fraction is None else memory_fraction,
                           workspace=self._workspace, total_memory=total_memory,
                           sub_batch_size=self.sub_batch_size)

    def _workspace_bytes(self, batch, n, device):
        if self._workspace is not None:
            return self._workspace(batch, n, device)
        if device.type == "cuda":
            try:
                return xsyevbatched_workspace_bytes(batch, n, device)
            except UnsupportedError:
                pass
        return None if batch * n * n > EIGH_UNQUERIED_ELEMENT_LIMIT else 4 * batch * n * n

    def _budget(self, device) -> float | None:
        if self.total_memory is None and device.type != "cuda":
            return None
        return self.memory_fraction * (self.total_memory or torch.cuda.get_device_properties(device).total_memory)

    def sub_batch(self, batch: int, n: int, device) -> tuple[int, str]:
        return self._sub_batch(batch, n, device)[:2]

    def _sub_batch(self, batch: int, n: int, device) -> tuple[int, str, bool]:
        device = torch.device(device)
        fixed = self.sub_batch_size(n)
        if fixed is not None:
            if type(fixed) is not int or fixed < 1:
                raise ConfigurationError(f"sub_batch_size({n}) must be a positive integer or None")
            budget = self._budget(device)
            reason = "per_matrix" if fixed == 1 else "fixed_sub_batch"
            if budget is None:
                return fixed, reason, False
            workspace = self._workspace_bytes(fixed, n, device)
            if workspace is None:
                raise UnsupportedError(f"the batched eigensolver rejects the fixed sub-batch of {fixed} {n}x{n} "
                                       "matrices")
            # Like one matrix in the halving path or a planner group of one: the minimum unit runs (recorded
            # ``over_budget``); a smaller sub-batch would change the eigenpairs.
            return fixed, reason, 8 * fixed * n * n + 4 * fixed * n + workspace > budget
        budget = self._budget(device)
        if budget is None:
            return batch, "whole_batch", False
        size, reason = batch, "whole_batch"
        while True:
            workspace = self._workspace_bytes(size, n, device)
            if workspace is None:
                reason = "solver_rejects_batch_size"
            elif 8 * size * n * n + 4 * size * n + workspace > budget:
                reason = "memory_bound"
            else:
                return size, reason, False
            if size == 1:
                if workspace is None:
                    raise UnsupportedError(f"the batched eigensolver rejects even one {n}x{n} matrix")
                return 1, reason, True
            size = (size + 1) // 2

    def __call__(self, grams: torch.Tensor):
        batch, n = int(grams.shape[0]), int(grams.shape[1])
        size, reason, over_budget = self._sub_batch(batch, n, grams.device)
        fixed = reason in ("per_matrix", "fixed_sub_batch")
        padded = 0
        if size == batch:
            values, vectors = self.solver(grams)
        else:
            unchecked = getattr(self.solver, "unchecked", None)
            values = vectors = None
            infos = []
            for start in range(0, batch, size):
                part = grams[start:start + size]
                count = int(part.shape[0])
                if fixed and count < size:
                    padded = size - count
                    part = torch.cat((part, torch.eye(n, dtype=grams.dtype, device=grams.device)
                                      .expand(padded, n, n)))
                part = part.contiguous()
                if unchecked is not None:
                    part_values, part_vectors, info = unchecked(part)
                    infos.append(info[:count])
                else:
                    part_values, part_vectors = self.solver(part)
                if values is None:
                    values = part_values.new_empty((batch, *part_values.shape[1:]))
                    vectors = part_vectors.new_empty((batch, *part_vectors.shape[1:]))
                values[start:start + count] = part_values[:count]
                vectors[start:start + count] = part_vectors[:count]
                del part, part_values, part_vectors
            if infos and bool((torch.cat(infos) != 0).any()):  # one host read for every sub-batch
                raise ConfigurationError("batched eigensolver did not converge")
        self.last = {"batch": batch, "n": n, "sub_batch": size, "solver_calls": -(-batch // size),
                     "reason": reason, **({"padded": padded} if padded else {}),
                     **({"over_budget": True} if over_budget else {})}
        if not fixed:
            self.last["sub_batch"] = min(size, batch)
        self.calls.append(self.last)
        return values, vectors


def select_eigh_solver(solver="auto", *, sub_batch_size=None):
    """Return ``(label, eigh)`` for stacked FP32 Grams; never silently degrade.

    ``auto`` prefers one batched ``torch.linalg.eigh`` call when the running
    Torch dispatches it to ``cusolverDnXsyevBatched`` (Torch >= 2.13), then the
    ctypes ``cusolverDnXsyevBatched`` binding, and otherwise raises
    UnsupportedError: a batched solve was requested and none is available.
    ``torch_loop`` is an explicit per-matrix diagnostic, never an automatic choice.
    ``eigh`` maps ``[b, n, n]`` to ascending values and *row* eigenvectors; the
    batched solvers are :class:`BoundedEigh` (automatic sub-batching of batches
    cuSOLVER rejects or the memory bound does not admit; same solver, recorded).
    ``sub_batch_size(n)`` replaces the default fixed sub-batch policy
    (:func:`default_eigh_sub_batch`: one matrix per call from n = 4096), e.g.
    ``lambda n: 1`` makes every eigensolve independent of its batch at any n.
    """
    if solver not in EIGH_SOLVERS:
        raise ConfigurationError(f"eigh solver must be one of {EIGH_SOLVERS}")
    from . import cusolver_eigh
    if solver in ("auto", "torch_batched") and torch_batched_eigh_is_xsyevbatched():
        label = "torch_linalg_eigh_batched_xsyevbatched"
        return label, BoundedEigh(label, _torch_rows_eigh, sub_batch_size=sub_batch_size)
    if solver == "torch_batched":
        raise UnsupportedError(
            f"Torch {torch.__version__} (CUDA {torch.version.cuda}) does not dispatch batched eigh to "
            "cusolverDnXsyevBatched; request cusolver_ctypes or upgrade Torch",
        )
    if solver in ("auto", "cusolver_ctypes"):
        if cusolver_eigh.available():
            label = "cusolver_xsyevbatched_ctypes"
            return label, BoundedEigh(label, cusolver_eigh.eigh_batched, sub_batch_size=sub_batch_size)
        raise UnsupportedError(
            f"a batched eigensolver was requested but is unavailable: Torch {torch.__version__} does not "
            "dispatch batched eigh to cusolverDnXsyevBatched and the ctypes cusolverDnXsyevBatched "
            "binding could not be loaded",
        )
    return "torch_eigh_loop_explicit", _torch_loop_eigh


def gram_bases(sources, ranks, *, eigh=None):
    """Top-``rank`` right singular bases of each ``[rows, n]`` source via one batched Gram eigh.

    Returns FP32 ``[rank, n]`` bases (descending energy, canonical signs not
    yet applied). ``eigh`` maps ``[b, n, n]`` to ``(values, row_vectors)``; the
    default uses ``torch.linalg.eigh`` (columns transposed to rows).
    """
    if not sources or len(sources) != len(ranks):
        raise ConfigurationError("batched Gram fitting needs one rank per source")
    channels = {int(x.shape[1]) for x in sources}
    if len(channels) != 1:
        raise ConfigurationError("one Gram batch must share the channel count")
    grams = torch.stack([mm(x.T, x) for x in sources])  # explicit IEEE FP32 (flag-independent)
    grams = (grams + grams.transpose(1, 2)) * 0.5
    if eigh is None:
        _, columns = torch.linalg.eigh(grams)
        vectors = columns.transpose(1, 2)
    else:
        _, vectors = eigh(grams)
    return [vectors[index, -rank:].flip(0) for index, rank in enumerate(ranks)]


class GPUFitter:
    def __init__(self, *, gpu_id, gpu_uuid, method="randomized", max_workspace_bytes=2 * 1024**3,
                 seed=0, power_iterations=2, batch_size=60, side_stream=False, shared=False,
                 eigh_solver="auto"):
        if type(gpu_id) is not int or gpu_id < 0 or not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
            raise ConfigurationError("GPU fitting requires an explicit visible device index and UUID")
        if method not in GPU_FIT_METHODS:
            raise ConfigurationError("unsupported GPU fitting backend")
        if type(max_workspace_bytes) is not int or max_workspace_bytes <= 0:
            raise ConfigurationError("GPU fitting workspace budget must be positive")
        if type(seed) is not int or seed < 0 or type(power_iterations) is not int or not 0 <= power_iterations <= 8:
            raise ConfigurationError("invalid GPU fitting seed or iteration count")
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ConfigurationError("GPU fitting batch size must be an integer in [1, 256]")
        self.gpu_id, self.gpu_uuid, self.method = gpu_id, gpu_uuid, method
        self.max_workspace_bytes, self.seed, self.power_iterations = max_workspace_bytes, seed, power_iterations
        self.batch_size = batch_size
        if type(side_stream) is not bool:
            raise ConfigurationError("side_stream must be boolean")
        if type(shared) is not bool:
            raise ConfigurationError("shared must be boolean")
        if eigh_solver not in EIGH_SOLVERS:
            raise ConfigurationError(f"eigh_solver must be one of {EIGH_SOLVERS}")
        self.eigh_solver = eigh_solver
        # ``shared`` lets CPU encode-pool workers call ``fit``: calls serialize on a lock and
        # run on the fitter's private CUDA stream, so they never touch the native forward stream.
        self.shared = shared
        self.side_stream = side_stream or shared
        self._stream = None
        self._lock = threading.Lock()
        self.owner = (os.getpid(), threading.get_ident())
        self._prefit: dict = {}
        self.prefit_stats = {"batches": 0, "matrices": 0, "hits": 0, "misses": 0, "solver": None}

    def adopt_current_thread(self):
        """Move ownership to the calling thread (e.g. a layer-streaming pipeline); not while fitting."""
        self.clear_prefit()
        self.owner = (os.getpid(), threading.get_ident())

    def _check_owner(self):
        if self.shared:
            if self.owner[0] != os.getpid():
                raise StateError("a shared GPU fitter cannot cross processes")
            return
        if self.owner != (os.getpid(), threading.get_ident()):
            raise StateError("GPU fitting must run on its configured owner thread, not the CPU encoder pool")

    def _device(self):
        if self.gpu_id >= torch.cuda.device_count():
            raise UnsupportedError("GPU fitting device is unavailable")
        device = torch.device("cuda", self.gpu_id)
        actual = str(torch.cuda.get_device_properties(device).uuid).removeprefix("GPU-")
        if actual != self.gpu_uuid.removeprefix("GPU-"):
            raise ConfigurationError("GPU fitting device differs from the granted UUID")
        return device

    @staticmethod
    def _identity(tensor, context, rank):
        return (
            tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype, str(tensor.device), tensor._version, context, rank,
        )

    @torch.no_grad()
    def prefit(self, items):
        """Batch-solve bases for ``{key: (tensor, context, rank)}`` ahead of per-matrix ``fit``.

        Only the ``gram_batched`` method batches; others return immediately.
        Each cached basis is consumed by the next ``fit`` of the *same, unmodified*
        tensor (storage, view, version, context and rank must match); the tensor
        is referenced until consumed or :meth:`clear_prefit`.
        """
        self._check_owner()
        self.clear_prefit()
        if self.method != "gram_batched" or not items:
            return
        prepared = []
        for tensor, context, rank in items.values():
            validate_matrix(tensor, context, lossy=True, check_values=False)
            validate_rank(rank, tuple(tensor.shape))
            prepared.append((tensor, context, rank))
        channels = max(int(t.shape[1]) for t, _, _ in prepared)
        rows = max(int(t.shape[0]) for t, _, _ in prepared)
        per_item = 3 * 4 * channels * channels
        batch = min(self.batch_size, max(1, (self.max_workspace_bytes - 8 * rows * channels) // per_item))
        if 8 * rows * channels + per_item > self.max_workspace_bytes:
            raise CapacityError("GPU batched fitting exceeds its declared logical workspace budget")
        device = self._device()
        label, solver = select_eigh_solver(self.eigh_solver)
        self.prefit_stats["solver"] = label
        with torch.cuda.device(device):
            for start in range(0, len(prepared), batch):
                group = prepared[start:start + batch]
                began = torch.cuda.Event(enable_timing=True)
                ended = torch.cuda.Event(enable_timing=True)
                began.record()
                sources = [validate_matrix(tensor.to(device), context).float() for tensor, context, _ in group]
                bases = gram_bases(sources, [rank for _, _, rank in group], eigh=solver)
                bases = [_canonical_signs(basis).to(dtype=torch.float16).contiguous() for basis in bases]
                ended.record()
                ended.synchronize()
                share = began.elapsed_time(ended) / 1e3 / len(group)
                del sources
                for (tensor, context, rank), basis in zip(group, bases):
                    if not bool(torch.isfinite(basis).all()):
                        raise ConfigurationError("fitted basis is not finite in stored FP16")
                    self._prefit[self._identity(tensor, context, rank)] = (tensor, basis, share)
                self.prefit_stats["batches"] += 1
                self.prefit_stats["matrices"] += len(group)

    def clear_prefit(self):
        self._prefit.clear()

    @torch.no_grad()
    def fit(self, tensor, context, rank, *, reuse_basis=None, reuse_threshold=None):
        self._check_owner()
        if self.shared:
            with self._lock:
                return self._fit(tensor, context, rank, reuse_basis=reuse_basis, reuse_threshold=reuse_threshold)
        return self._fit(tensor, context, rank, reuse_basis=reuse_basis, reuse_threshold=reuse_threshold)

    def _fit(self, tensor, context, rank, *, reuse_basis=None, reuse_threshold=None):
        # Values are scanned on the device copy below (fit_basis / projection), not on the host.
        validate_matrix(tensor, context, lossy=True, check_values=False)
        validate_rank(rank, tuple(tensor.shape))
        rows, channels = tensor.shape
        estimated = 32 * (rows * channels + channels * channels + rows * rank)
        if estimated > self.max_workspace_bytes:
            raise CapacityError("GPU fitting exceeds its declared logical workspace budget")
        device = self._device()
        if reuse_basis is not None or reuse_threshold is not None:
            raise UnsupportedError("explicit GPU solver comparisons currently require fresh bases")
        started = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        if self.side_stream and self._stream is None:
            self._stream = torch.cuda.Stream(device)
        with torch.cuda.device(device), (
            torch.cuda.stream(self._stream) if self.side_stream else contextlib.nullcontext()
        ):
            events[0].record()
            source = tensor.to(device)
            events[1].record()
            if self.method == "gram_batched":
                factors = self._fit_gram_batched(source, tensor, context, rank)
            else:
                factors = fit_basis(source, context, rank, method=self.method,
                                    seed=self.seed, power_iterations=self.power_iterations)
            events[2].record()
            basis = factors.basis.cpu()
            coefficients = factors.coefficients.cpu()
            events[3].record()
            events[3].synchronize()
        metrics = {
            **factors.metrics, "gpu_fitting_backend": self.method,
            "gpu_fitting_uuid": self.gpu_uuid, "gpu_fitting_workspace_estimate": estimated,
            "gpu_fitting_h2d_ms": events[0].elapsed_time(events[1]),
            "gpu_fitting_compute_ms": events[1].elapsed_time(events[2]),
            "gpu_fitting_d2h_ms": events[2].elapsed_time(events[3]),
            "gpu_fitting_wall_seconds": time.perf_counter() - started,
            "reuse_attempts": 0, "reuse_accepted": False, "reuse_rejections": 0,
            "reuse_reason": "gpu_fresh_basis", "reuse_seconds": 0.0,
            "total_factorization_seconds": time.perf_counter() - started,
            "canonical_basis_rank": rank, "view_rank": rank,
        }
        return Factors(basis=basis, coefficients=coefficients, rank=rank, metrics=metrics)

    def _fit_gram_batched(self, source, tensor, context, rank):
        """Use the prefit basis when available; otherwise a single-matrix Gram eigh."""
        started = time.perf_counter()
        cached = self._prefit.pop(self._identity(tensor, context, rank), None)
        if cached is not None:
            validate_matrix(source, context)
            self.prefit_stats["hits"] += 1
            basis, decomposition_seconds = cached[1], cached[2]
            algorithm = "batched_fp32_gram_eigh"
        else:
            self.prefit_stats["misses"] += 1
            decomposition_started = time.perf_counter()
            (basis,) = gram_bases([validate_matrix(source, context).float()], [rank])
            basis = _canonical_signs(basis).to(dtype=torch.float16).contiguous()
            torch.cuda.current_stream(source.device).synchronize()
            decomposition_seconds = time.perf_counter() - decomposition_started
            if not bool(torch.isfinite(basis).all()):
                raise ConfigurationError("fitted basis is not finite in stored FP16")
            algorithm = "fp32_gram_eigh"
        projection = project_onto_basis(source, basis, rank=rank, _condition_bound=fast_condition_bound(basis))
        return Factors(basis=basis, coefficients=projection.coefficients, rank=rank, metrics={
            "factorization_method": algorithm, "factorization_seed": self.seed,
            "sketch_rank": None, "power_iterations": 0, "decomposition_count": 1,
            "svd_count": 0, "eigh_count": 1, "qr_count": 0,
            "decomposition_seconds": decomposition_seconds,
            "projection_count": 1, "projection_seconds": projection.seconds,
            "projection_relative_residual": projection.relative_residual,
            "basis_condition_bound": projection.condition_bound,
            "basis_condition_method": "fp32_neumann_upper_bound",
            "zero_input": projection.zero_input,
            "fit_seconds": time.perf_counter() - started,
            "prefit_batched": cached is not None,
        })

    @torch.no_grad()
    def plan_chunk(self, config, view, sources, contexts, bindings, parent_rows, *, chunk_id, source_ids,
                   orthogonalizer="householder", max_working_bytes=None):
        """Latest-method chunk plan (sketch, predict, batched Gram eigh, verify/grow/trim,
        certify/share, augment, refit) on this fitter's granted device and owner thread.

        The batched eigensolver follows ``eigh_solver`` (fail loudly when a batched
        solve is unavailable). ``max_working_bytes`` bounds the planner's working set
        (consecutive matrix groups; identical plans). Returns the :class:`reelcache.rank_method.ChunkPlan`.
        """
        from .rank_method import LatestMethodPlanner
        self._check_owner()
        device = self._device()
        label, solver = select_eigh_solver(self.eigh_solver)
        planner = LatestMethodPlanner(config, view, device=device, eigh=solver, eigh_label=label,
                                      orthogonalizer=orthogonalizer, max_working_bytes=max_working_bytes)
        if self.side_stream and self._stream is None:
            self._stream = torch.cuda.Stream(device)
        scope = self._lock if self.shared else contextlib.nullcontext()
        with scope, torch.cuda.device(device), (
            torch.cuda.stream(self._stream) if self.side_stream else contextlib.nullcontext()
        ):
            plan = planner.plan(sources, contexts, bindings, parent_rows, chunk_id=chunk_id, source_ids=source_ids)
        self.prefit_stats["solver"] = label
        return plan
