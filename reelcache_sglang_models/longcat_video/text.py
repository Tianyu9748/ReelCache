"""Native LongCat-Video prompt cleaning and tokenization arguments (pure Python)."""

from __future__ import annotations

import html

from .spec import MAX_SEQUENCE_LENGTH


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as exc:  # explicit: a different cleaning would change the prompt tokens
        raise ImportError(f"LongCat-Video prompt cleaning needs the '{module}' package (native uses it)") from exc


def basic_clean(text: str) -> str:
    ftfy = _require("ftfy")
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    regex = _require("regex")
    return regex.sub(r"\s+", " ", text).strip()


def prompt_clean(text: str) -> str:
    """``pipeline_longcat_video.prompt_clean``: ftfy fix, double HTML unescape, whitespace collapse."""
    return whitespace_clean(basic_clean(text))


def tokenizer_kwargs(max_sequence_length: int = MAX_SEQUENCE_LENGTH) -> dict:
    """``_get_t5_prompt_embeds`` tokenizer call (padding to 512, truncation, special tokens, mask)."""
    return {"padding": "max_length", "max_length": max_sequence_length, "truncation": True,
            "add_special_tokens": True, "return_attention_mask": True, "return_tensors": "pt"}
