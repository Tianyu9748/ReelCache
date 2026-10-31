//! Accel track C (C3): Rust implementations of the archival host hot spots that remain after the Python fast paths.
//!
//! Every function returns exactly what the Python reference returns (property-tested in
//! tests/test_accel_c_native.py) and runs its work with the GIL released (`py.allow_threads`).
//! * `start_codes(data)`: every offset of `00 00 01` (the matches of `_hevc._START_CODE.finditer`).
//! * `nal_units(data)`: `_hevc.nal_units` (offsets, lengths, NAL types, first-slice flags) with the same checks; a
//!   failed check raises `ValueError` carrying the reference's `IntegrityError` message (the caller re-raises it).
//! * `unpack_tile(canvas, values, ...)`: `codecs._unpack_tile` (the per-cell `_hevc._unmosaic` copies of every segment)
//!   from a C-contiguous uint8 canvas `[pictures, planes, height, width]` into a C-contiguous uint8
//!   `[frames, h, w, rank]` array.

use memchr::memmem;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn scan(data: &[u8]) -> Vec<usize> {
    memmem::find_iter(data, b"\x00\x00\x01").collect()
}

#[pyfunction]
fn start_codes(py: Python<'_>, data: &[u8]) -> Vec<usize> {
    py.allow_threads(|| scan(data))
}

type Unit = (usize, usize, u32, Option<bool>);

fn units(data: &[u8]) -> Result<Vec<Unit>, &'static str> {
    let positions = scan(data);
    let mut starts: Vec<(usize, usize)> = Vec::with_capacity(positions.len());
    for position in positions {
        let header = position + 3;
        let start = if position > 0 && data[position - 1] == 0 { position - 1 } else { position };
        starts.push((start, header));
    }
    if starts.is_empty() || starts[0].0 != 0 {
        return Err("HEVC stream does not begin with an Annex-B start code");
    }
    let total = data.len();
    let mut out = Vec::with_capacity(starts.len());
    for index in 0..starts.len() {
        let (start, header) = starts[index];
        let end = if index + 1 < starts.len() { starts[index + 1].0 } else { total };
        if header + 2 > end {
            return Err("truncated HEVC NAL unit header");
        }
        let (first, second) = (data[header], data[header + 1]);
        if first & 0x80 != 0 || second & 0x07 == 0 {
            return Err("invalid HEVC NAL unit header");
        }
        let nal_type = ((first >> 1) & 0x3F) as u32;
        let mut first_slice = None;
        if nal_type < 32 {
            if header + 2 >= end {
                return Err("truncated HEVC slice segment");
            }
            first_slice = Some(data[header + 2] & 0x80 != 0);
        }
        out.push((start, end - start, nal_type, first_slice));
    }
    Ok(out)
}

#[pyfunction]
fn nal_units(py: Python<'_>, data: &[u8]) -> PyResult<Vec<Unit>> {
    py.allow_threads(|| units(data)).map_err(|message| PyValueError::new_err(message))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn unpack_tile(
    py: Python<'_>,
    canvas: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    canvas_shape: (usize, usize, usize, usize),
    first_picture: usize,
    x0: usize,
    y0: usize,
    frames: usize,
    height: usize,
    width: usize,
    rank: usize,
    columns: usize,
    rows: usize,
    planes: usize,
    segments: usize,
) -> PyResult<()> {
    let source = PyBuffer::<u8>::get(canvas)?;
    let target = PyBuffer::<u8>::get(values)?;
    let (pictures, canvas_planes, canvas_height, canvas_width) = canvas_shape;
    if !source.is_c_contiguous() || !target.is_c_contiguous() || target.readonly() {
        return Err(PyValueError::new_err("unpack_tile needs C-contiguous buffers and a writable target"));
    }
    let slots = columns * rows * planes;
    if source.len_bytes() != pictures * canvas_planes * canvas_height * canvas_width
        || target.len_bytes() != frames * height * width * rank
        || slots == 0
        || segments * slots < rank
        || planes > canvas_planes
        || first_picture + segments * frames > pictures
        || y0 + rows * height > canvas_height
        || x0 + columns * width > canvas_width
    {
        return Err(PyValueError::new_err("unpack_tile geometry is outside its buffers"));
    }
    let src_addr = source.buf_ptr() as usize;
    let dst_addr = target.buf_ptr() as usize;
    py.allow_threads(move || {
        // SAFETY: both buffers stay exported (PyBuffer held) and were bounds-checked above; the target is writable.
        let src = unsafe { std::slice::from_raw_parts(src_addr as *const u8, pictures * canvas_planes * canvas_height * canvas_width) };
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_addr as *mut u8, frames * height * width * rank) };
        let plane_stride = canvas_height * canvas_width;
        let picture_stride = canvas_planes * plane_stride;
        for segment in 0..segments {
            let first = segment * slots;
            let count = slots.min(rank - first);
            for frame in 0..frames {
                let picture = first_picture + segment * frames + frame;
                for y in 0..height {
                    let row_base = (frame * height + y) * width * rank;
                    for index in 0..count {
                        let plane = index % planes;
                        let cell = index / planes;
                        let (row, column) = (cell / columns, cell % columns);
                        let offset = picture * picture_stride + plane * plane_stride
                            + (y0 + row * height + y) * canvas_width + x0 + column * width;
                        let line = &src[offset..offset + width];
                        let channel = first + index;
                        for (x, value) in line.iter().enumerate() {
                            dst[row_base + x * rank + channel] = *value;
                        }
                    }
                }
            }
        }
    });
    drop(source);
    drop(target);
    Ok(())
}

#[pymodule]
fn reelcache_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(start_codes, module)?)?;
    module.add_function(wrap_pyfunction!(nal_units, module)?)?;
    module.add_function(wrap_pyfunction!(unpack_tile, module)?)?;
    module.add("__version__", "0.1.0")?;
    Ok(())
}
