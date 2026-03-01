from __future__ import annotations

"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import os
import math
import re
import sys
from typing import Any, Dict, List, Tuple
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import accelerate
import hydra
import numpy as np
import torch
import torch.distributed as dist
from lm_eval.api.model import LM
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import make_table
from omegaconf import DictConfig
from tqdm import tqdm
import imageio.v2 as imageio
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import ModelOutput

from datasets import Dataset
from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs

# visualize the intermediate samples, create a gif            
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from typing import List, Tuple, Optional, Callable


def _find_monospace_font_path() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/Library/Fonts/Courier New.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return font_manager.findfont(font_manager.FontProperties(family="monospace"))


def _to_gif_palette(im: Image.Image) -> Image.Image:
    # Adaptive 256-color palette + dithering improves GIF text/box quality substantially.
    return im.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.FLOYDSTEINBERG)


def wrap_by_pixels(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[Tuple[str, int, bool]]:
    """
    Returns [(line_text, global_char_start_index_in_original_text)].
    Character indexing is with respect to `text`.
    """
    paragraphs = text.split("\n")
    lines_with_idx: List[Tuple[str, int, bool]] = []
    global_idx = 0

    for p_i, para in enumerate(paragraphs):
        has_nl_after_para = (p_i < len(paragraphs) - 1)
        if para == "":
            lines_with_idx.append(("", global_idx, has_nl_after_para))
            if has_nl_after_para:
                global_idx += 1
            continue

        words = para.split(" ")
        cur = ""
        cur_start = global_idx
        running_idx = global_idx

        for w_i, w in enumerate(words):
            token = w if w_i == 0 else (" " + w)
            candidate = cur + token

            if draw.textlength(candidate, font=font) <= max_width or cur == "":
                cur = candidate
            else:
                lines_with_idx.append((cur, cur_start, False))
                cur = w
                cur_start = running_idx + (1 if w_i != 0 else 0)

            running_idx += len(token)

        lines_with_idx.append((cur, cur_start, has_nl_after_para))

        global_idx = running_idx + (1 if has_nl_after_para else 0)

    return lines_with_idx

def strings_to_gif_oldstyle(
    tokenizer,
    tokens,                         # iterable of sequences of token ids, one per frame
    highlight_token_indices=None,    # per-frame token indices to highlight (can be non-contiguous)
    out_path="strings.gif",
    size=(400, 2200),

    # Style knobs
    scale=2,
    downscale_output=True,
    padding=5,
    line_spacing=10,
    font_size=25,

    # Box styling
    box_fill=(232, 210, 109),
    underlay_fill=(255, 255, 255),  # NEW: white underlay fill (same as background)
    text_color=(0, 0, 0),
    latest_outline=(0, 0, 0),
    border_width_thick=0,
    border_width_thin=1,
    corner_radius=6,
    dot_spacing=10,
    dot_radius=1,
    # Token box separability (NEW)
    # - token_box_inset_px: shrinks boxes tighter around glyphs
    # - token_box_gap_px: adds visual "air" between adjacent token boxes by shrinking each side
    token_box_inset_px=1,
    token_box_gap_px=2,

    # Mask handling
    mask_token="<|fim_middle|>",

    # Timing
    frame_ms=100,
    loop=0,
    block_diffusion=False,
):
    """
    Per-frame visualization of decoded text with per-token highlight boxes.

    - `highlight_token_indices[i]` can be list/set/1D tensor of token indices for frame i.
      If None for a frame, the entire line is boxed (back-compat behavior).
    - Boxes are drawn *per token span* (no merging), even if adjacent.
    - `mask_token` positions are replaced with final-frame token IDs for layout, but drawn invisible.

    Notes on vertical alignment:
    - PIL draws text at a baseline y; we compute ascent/descent and place the box around that baseline.
    - `box_offset` moves boxes downward; increase it to push boxes down further.
    """
    font_path = _find_monospace_font_path()

    s = max(1, int(scale))
    W, H = size
    render_W = W * s if downscale_output else W
    render_H = H * s if downscale_output else H

    font = ImageFont.truetype(font_path, int(font_size * s))

    decode_kwargs = dict(
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    bg_color = (255, 255, 255)


    # Resolve mask token id (may be None if tokenizer doesn't know it)
    try:
        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    except Exception:
        mask_id = None

    # Normalize final frame ids once for GT indexing
    gt_ids: Optional[List[int]] = None
    if tokens:
        last = tokens[-1]
        if isinstance(last, int):
            gt_ids = [int(last)]
        else:
            try:
                gt_ids = [int(x) for x in last]
            except TypeError:
                gt_ids = [int(last)]

    def draw_box(
        draw,
        bbox,
        fill,
        outline=None,
        width=1,
        radius=8,
    ):
        # No dotted borders anywhere. (All boxes are solid fill; optional solid outline.)
        draw.rounded_rectangle(bbox, radius=radius, fill=fill)
        if outline is not None and width > 0:
            draw.rounded_rectangle(bbox, radius=radius, outline=outline, width=width)

    def draw_text_only_frame(s_full: str) -> Image.Image:
        im = Image.new("RGB", (render_W, render_H), color=bg_color)
        draw = ImageDraw.Draw(im)

        wrapped = wrap_by_pixels(s_full, draw, font, max_width=text_area_w)

        x0 = int(padding * s)
        y0 = int(padding * s)
        yy = y0

        for line, _, _ in wrapped:
            if line == "":
                bbox = draw.textbbox((0, 0), "Ag", font=font)
                lh = bbox[3] - bbox[1]
                yy += lh + int(line_spacing * s)
                continue

            # IMPORTANT: match intermediate frames (same baseline convention)
            draw.text((x0, yy), line, fill=text_color, font=font)

            bbox = draw.textbbox((0, 0), line, font=font)
            lh = bbox[3] - bbox[1]
            yy += lh + int(line_spacing * s)

        if downscale_output:
            im = im.resize((W, H), Image.LANCZOS)

        return im

    def save_frames_to_mp4(frames_rgb, out_path, fps, last_frame_hold_s=0.0):
        if not out_path.lower().endswith(".mp4"):
            out_path = os.path.splitext(out_path)[0] + ".mp4"

        with imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        ) as writer:
            for im in frames_rgb:
                writer.append_data(np.asarray(im, dtype=np.uint8))

            if last_frame_hold_s and last_frame_hold_s > 0:
                n_hold = int(round(last_frame_hold_s * fps))
                if n_hold > 0:
                    last = np.asarray(frames_rgb[-1], dtype=np.uint8)
                    for _ in range(n_hold):
                        writer.append_data(last)

    def draw_line_with_invisible_spans(
        draw: ImageDraw.ImageDraw,
        x_start: int,
        y: int,
        line_text: str,
        line_char_start: int,
        invisible_spans: List[Tuple[int, int]],
        highlight_bg_spans_global: Optional[List[Tuple[int, int, Tuple[int, int, int]]]],
        font: ImageFont.ImageFont,
        fg,
        bg_default,        # usually bg_color (white)
    ):
        if not line_text:
            return
        if not invisible_spans:
            draw.text((x_start, y), line_text, fill=fg, font=font)
            return

        line_char_end = line_char_start + len(line_text)

        overlaps: List[Tuple[int, int]] = []
        for a, b in invisible_spans:
            lo = max(a, line_char_start)
            hi = min(b, line_char_end)
            if hi > lo:
                overlaps.append((lo - line_char_start, hi - line_char_start))

        if not overlaps:
            draw.text((x_start, y), line_text, fill=fg, font=font)
            return

        overlaps.sort()
        merged: List[Tuple[int, int]] = []
        for a, b in overlaps:
            if not merged or a > merged[-1][1]:
                merged.append((a, b))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))

        x = x_start
        cur = 0

        def _w(seg: str) -> int:
            return int(draw.textlength(seg, font=font))

        def _fill_for_invisible(global_a: int, global_b: int):
            """
            Invisible text should match whatever is *behind* it:
              - if it falls under a highlight box => match that box's (effective) RGB
              - otherwise => bg_default (white)
            highlight_bg_spans_global contains (a, b, rgb) spans in *global* char coords.
            """
            if not highlight_bg_spans_global:
                return [(global_a, global_b, bg_default)]

            # Collect intersections with highlight spans, keeping their colors.
            inters: List[Tuple[int, int, Tuple[int, int, int]]] = []
            for ha, hb, hc in highlight_bg_spans_global:
                lo = max(global_a, ha)
                hi = min(global_b, hb)
                if hi > lo:
                    inters.append((lo, hi, hc))
            if not inters:
                return [(global_a, global_b, bg_default)]

            # Split [global_a, global_b) into colored segments + background gaps.
            # Note: highlight spans may overlap; we keep later spans in list order only where they apply.
            # For our use (per-token, non-overlapping), this is sufficient.
            inters.sort(key=lambda x: (x[0], x[1]))
            pieces: List[Tuple[int, int, Tuple[int, int, int]]] = []
            p = global_a
            for a, b, c in inters:
                if a > p:
                    pieces.append((p, a, bg_default))
                pieces.append((a, b, c))
                p = max(p, b)
            if p < global_b:
                pieces.append((p, global_b, bg_default))
            return pieces

        for a, b in merged:
            if a > cur:
                seg = line_text[cur:a]
                draw.text((x, y), seg, fill=fg, font=font)
                x += _w(seg)
            seg = line_text[a:b]
            # Invisible text should match what's behind it:
            # - inside highlighted region => highlight box color (incl. alpha composited)
            # - otherwise => bg_default (white)
            global_a = line_char_start + a
            global_b = line_char_start + b
            # Draw the invisible segment possibly split by highlight overlap.
            for pa, pb, fillc in _fill_for_invisible(global_a, global_b):
                # pa/pb are global; map to local indices within this line_text
                la = pa - line_char_start
                lb = pb - line_char_start
                sub = line_text[la:lb]
                draw.text((x, y), sub, fill=fillc, font=font)
                x += _w(sub)
            cur = b

        if cur < len(line_text):
            seg = line_text[cur:]
            draw.text((x, y), seg, fill=fg, font=font)

    def _span_covers(spans: List[Tuple[int, int]], idx: int) -> bool:
        """
        spans are half-open intervals [a, b).
        True iff idx lies within any span.
        """
        return any(a <= idx < b for (a, b) in spans)

    def _should_box_newline(text: str, spans: List[Tuple[int, int]], nl_idx: int) -> bool:
        """
        Only draw a newline box when the highlighted token is *a newline token* at that point,
        not when the newline is merely a trailing character of a token that also contains a
        visible glyph (e.g. ':\n\n' should NOT highlight the newlines, only ':').

        Rules:
        - newline must be covered by at least one highlighted span
        - allow boxing if the newline is the *first* char of the span (token starts with '\n')
        - also allow boxing for consecutive newlines within the same span (prev char is '\n')
        - otherwise (prev char is a visible glyph inside the same span), suppress.
        """
        if nl_idx < 0 or nl_idx >= len(text):
            return False
        if text[nl_idx] != "\n":
            return False

        for a, b in spans:
            if a <= nl_idx < b:
                # NEW: if this token-span contains ANY non-newline visible character,
                # then do not draw newline boxes for it. This fixes cases like ":\n\n"
                # where the token includes a printable glyph plus trailing newlines:
                # only the printable glyph should be highlighted.
                #
                # NOTE: we only treat '\n' as "newline". If your tokenizer can emit '\r\n',
                # you may want to treat '\r' similarly.
                if any(ch != "\n" for ch in text[a:b]):
                    return False

                # Newline is the first character emitted by this token-span.
                if nl_idx == a:
                    return True
                # Handle "\n\n" (or longer) emitted by one token-span:
                # box the later newline(s) where the previous character is also '\n'.
                if nl_idx > 0 and text[nl_idx - 1] == "\n" and a <= (nl_idx - 1) < b:
                    return True
        return False

    def _is_consecutive_newline_in_same_span(
        text: str, spans: List[Tuple[int, int]], nl_idx: int
    ) -> bool:
        """
        If the highlighted span covers multiple consecutive '\n' characters (e.g. a single token
        decodes to '\n\n'), suppress boxing the earlier newline(s) so we only box the new empty line.

        Concretely: when deciding whether to draw a box for the newline at `nl_idx`, if:
          - text[nl_idx] is '\n'
          - text[nl_idx+1] is also '\n'
          - both indices are covered by the highlight spans
        then we skip boxing at nl_idx and let the next line's newline get boxed instead.
        """
        if nl_idx < 0 or nl_idx >= len(text):
            return False
        if text[nl_idx] != "\n":
            return False
        nxt = nl_idx + 1
        if nxt < len(text) and text[nxt] == "\n":
            return _span_covers(spans, nl_idx) and _span_covers(spans, nxt)
        return False

    def _newline_attached_to_prev_visible_char(
        text: str, spans: List[Tuple[int, int]], nl_idx: int
    ) -> bool:
        """
        If the highlighted span includes a newline at `nl_idx` *and* also includes the immediately
        preceding character (e.g. token delta is ':\n'), then we should NOT box the newline.
        The token's visible character(s) will already be boxed by the per-token span logic.
        """
        if nl_idx <= 0 or nl_idx >= len(text):
            return False
        if text[nl_idx] != "\n":
            return False
        # If the token span covers the newline and also covers the previous character, treat the
        # newline as "attached" to the prior glyph and suppress newline boxing.
        return (
            _span_covers(spans, nl_idx)
            and _span_covers(spans, nl_idx - 1)
            and (text[nl_idx - 1] != "\n")
        )

    frames_rgb: List[Image.Image] = []
    text_area_w = render_W - 2 * int(padding * s)

    # Baseline-consistent box placement:
    # yy is baseline. Text occupies [yy-ascent, yy+descent]. Box is padded and shifted down by box_offset.
    ascent, descent = font.getmetrics()
    box_offset = int(0.35 * ascent)  # increase => boxes move further down

    # NEW: enforce a minimum highlight-box height so punctuation/short glyph lines
    # don't produce "squashed" boxes.
    #
    # PIL's textbbox can be quite short for lines that only contain low-ink glyphs
    # (e.g., "." or ","), which in turn makes highlight boxes too short.
    _min_box_h = int(max(6 * s, 0.9 * (ascent + descent)))  # render-space pixels

    def _ensure_min_box_height(
        y_top: int,
        y_bot: int,
        min_h: int,
        canvas_h: int,
    ) -> Tuple[int, int]:
        """Expand (y_top, y_bot) to be at least min_h tall, clamped to [0, canvas_h]."""
        h = y_bot - y_top
        if h >= min_h:
            return y_top, y_bot
        extra = min_h - h
        # Expand symmetrically; if we hit bounds, shift as needed.
        new_top = y_top - (extra // 2)
        new_bot = y_bot + (extra - (extra // 2))
        if new_top < 0:
            new_bot = min(canvas_h, new_bot + (-new_top))
            new_top = 0
        if new_bot > canvas_h:
            new_top = max(0, new_top - (new_bot - canvas_h))
            new_bot = canvas_h
        return new_top, new_bot

    # NEW: convert separability knobs to render-space pixels
    # NOTE: these act by shrinking the box horizontally, which both (a) tightens the box
    # around the token glyphs and (b) increases the visible gap between adjacent boxes.
    _token_inset = int(max(0, token_box_inset_px) * s)
    _token_gap = int(max(0, token_box_gap_px) * s)
    # Each token box is shrunk by half the gap on each side, plus inset.
    _shrink_left = _token_inset + (_token_gap // 2)
    _shrink_right = _token_inset + (_token_gap - (_token_gap // 2))
    # Prevent boxes collapsing to nothing for very thin glyph spans.
    _min_token_box_w = int(4 * s)

    prev_decoded_global = ""

    def _gradient_alpha_for_k(k: int, n: int) -> int:
        """
        Alpha gradient for gold token boxes in non-block diffusion:
        - fully opaque at k=0 (start_tok)
        - very transparent at k=n-1 (start_tok+8)
        """
        if n <= 1:
            return 255
        a0 = 255  # start: no transparency
        a1 = 25   # end: very transparent
        t = float(k) / float(n - 1)
        return int(round(a0 * (1.0 - t) + a1 * t))

    def _flatten_rgba_on_white(im_rgba: Image.Image) -> Image.Image:
        """
        Pillow's convert('RGB') drops alpha without compositing. For semi-transparent
        fills to actually appear lighter, alpha-composite onto a white background first.
        """
        if im_rgba.mode != "RGBA":
            im_rgba = im_rgba.convert("RGBA")
        white = Image.new("RGBA", im_rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, im_rgba).convert("RGB")

    def _blend_over_white(rgb: Tuple[int, int, int], alpha: int) -> Tuple[int, int, int]:
        """
        Return the effective RGB when an RGBA fill is composited over a white background.
        This is what the viewer actually sees after we later flatten RGBA->RGB.
        """
        a = max(0, min(255, int(alpha)))
        r, g, b = rgb
        # out = a*src + (1-a)*white
        rr = (a * r + (255 - a) * 255) // 255
        gg = (a * g + (255 - a) * 255) // 255
        bb = (a * b + (255 - a) * 255) // 255
        return (int(rr), int(gg), int(bb))

    for i, t in enumerate(tokens):
        # Normalize token ids to list[int]
        if isinstance(t, int):
            ids = [int(t)]
        else:
            try:
                ids = [int(x) for x in t]
            except TypeError:
                ids = [int(t)]

        # Collect per-token highlight indices for this frame
        frame_highlight: Optional[List[int]] = None
        if highlight_token_indices is not None and i < len(highlight_token_indices):
            hi = highlight_token_indices[i]
            if hi is None:
                frame_highlight = None
            elif torch.is_tensor(hi):
                frame_highlight = hi.detach().cpu().tolist()
            elif isinstance(hi, (set, tuple, list, np.ndarray)):
                frame_highlight = [int(x) for x in hi]
            else:
                frame_highlight = [int(hi)]

        # NEW: compute a contiguous "underlay" span (light grey box) for non-block diffusion.
        # It starts at:
        #   min(first mask index in accumulated_samples, highlight_token_indices[0])
        # and ends 8 token positions later.
        #
        # Interpretation here (per-frame):
        # - start_tok := min(first mask index, first highlighted index), clamped >= 0
        # - "accumulated_samples" := the current frame's token-id list `ids`
        # - "first mask index" := first position where the token id equals `mask_id`
        # - "highlight_token_indices[0]" := the first highlighted token index for this frame
        #   (i.e., frame_highlight[0])
        underlay_token_span: Optional[Tuple[int, int]] = None  # [start_tok, end_tok_exclusive)
        if not block_diffusion:
            first_mask_idx: Optional[int] = None
            if mask_id is not None:
                for _j, _tid in enumerate(ids):
                    if int(_tid) == int(mask_id):
                        first_mask_idx = _j
                        break

            first_hi_idx: Optional[int] = None
            if frame_highlight is not None and len(frame_highlight) > 0:
                first_hi_idx = int(frame_highlight[0])

            start_candidates: List[int] = []
            if first_mask_idx is not None:
                start_candidates.append(int(first_mask_idx))
            if first_hi_idx is not None:
                start_candidates.append(int(first_hi_idx))

            if len(start_candidates) > 0:
                start_tok = max(0, min(start_candidates))
                end_tok_excl = start_tok + 9  # start_tok .. start_tok+8 (inclusive)
                underlay_token_span = (start_tok, end_tok_excl)

        # Build RAW text deltas + RAW per-token char spans
        pieces: List[str] = []
        invisible_spans: List[Tuple[int, int]] = []        # RAW spans to render invisible
        token_char_ranges_raw: List[Tuple[int, int]] = []  # RAW spans per token

        pos = 0
        running_ids: List[int] = []
        prev_decoded = ""

        for j, tid in enumerate(ids):
            use_id = int(tid)
            made_invisible = False

            if (
                mask_id is not None
                and use_id == mask_id
                and gt_ids is not None
                and j < len(gt_ids)
            ):
                use_id = gt_ids[j]
                made_invisible = True

            running_ids.append(use_id)

            decoded = tokenizer.decode(running_ids, **decode_kwargs)

            # Delta: what new text appeared by adding this token?
            if decoded.startswith(prev_decoded):
                txt = decoded[len(prev_decoded):]
            else:
                k = 0
                m = min(len(decoded), len(prev_decoded))
                while k < m and decoded[k] == prev_decoded[k]:
                    k += 1
                txt = decoded[k:]

            prev_decoded = decoded
            prev_decoded_global = decoded  # keep last decoded for final frame

            # Drop intermediate Unicode replacement chars from partial UTF-8 sequences.
            txt = txt.replace("\ufffd", "")

            start = pos
            end = pos + len(txt)
            token_char_ranges_raw.append((start, end))
            pieces.append(txt)

            if made_invisible and txt:
                invisible_spans.append((start, end))

            pos = end

        raw_full = "".join(pieces)

        # Cosmetic replacement
        s_full = raw_full.replace("✅", "✔")

        token_char_ranges = token_char_ranges_raw
        invisible_spans_vis = invisible_spans

        # Determine which tokens to highlight:
        # - block diffusion: use provided highlight_token_indices
        # - non-block diffusion: force highlight of tokens start_tok..start_tok+8 with alpha gradient
        gradient_tok_to_alpha: Dict[int, int] = {}
        if (not block_diffusion) and (underlay_token_span is not None):
            st, en = underlay_token_span
            n = max(0, en - st)
            for k in range(n):
                gradient_tok_to_alpha[st + k] = _gradient_alpha_for_k(k, n)

        # Build per-token spans to highlight (VISIBLE spans; DO NOT MERGE)
        # Store token index as well so we can apply per-token alpha in non-block diffusion.
        highlight_token_spans: Optional[List[Tuple[int, int, int]]] = None  # (tok_idx, a, b)
        if block_diffusion:
            if frame_highlight is not None:
                spans3: List[Tuple[int, int, int]] = []
                for tok_idx in frame_highlight:
                    if 0 <= tok_idx < len(token_char_ranges):
                        a, b = token_char_ranges[tok_idx]
                        if b > a:
                            spans3.append((tok_idx, a, b))
                highlight_token_spans = spans3
        else:
            if gradient_tok_to_alpha:
                spans3 = []
                for tok_idx in sorted(gradient_tok_to_alpha.keys()):
                    if 0 <= tok_idx < len(token_char_ranges):
                        a, b = token_char_ranges[tok_idx]
                        if b > a:
                            spans3.append((tok_idx, a, b))
                highlight_token_spans = spans3

        # NEW: map the underlay token-span (token indices) to a single char-span (contiguous)
        underlay_char_span: Optional[Tuple[int, int]] = None
        if underlay_token_span is not None:
            start_tok, end_tok_excl = underlay_token_span
            # clamp to available token ranges
            start_tok = (
                max(0, min(start_tok, len(token_char_ranges) - 1)) if token_char_ranges else 0
            )
            end_tok_excl = max(0, min(end_tok_excl, len(token_char_ranges)))
            if token_char_ranges and end_tok_excl > start_tok:
                a = token_char_ranges[start_tok][0]
                b = token_char_ranges[end_tok_excl - 1][1]
                if b > a:
                    underlay_char_span = (a, b)

        # Use RGBA so non-block diffusion highlight boxes can have alpha gradients.
        im = Image.new("RGBA", (render_W, render_H), color=bg_color + (255,))
        draw = ImageDraw.Draw(im, "RGBA")

        wrapped = wrap_by_pixels(s_full, draw, font, max_width=text_area_w)

        # Precompute line heights (for line spacing progression)
        line_heights = []
        for line, _start, _nl_after in wrapped:
            bbox = draw.textbbox((0, 0), line if line else "Ag", font=font)
            line_heights.append(bbox[3] - bbox[1])

        x0 = int(padding * s)
        y0 = int(padding * s)

        yy = y0
        min_nl_box_px = int(10 * s)  # minimum width for newline box (tweak)
        _min_nl_box_px = max(min_nl_box_px, _min_token_box_w)

        for (line, line_start, hard_nl_after), lh in zip(wrapped, line_heights):
            line_end = line_start + len(line)

            def w(sub: str) -> int:
                return int(draw.textlength(sub, font=font))

            # Compute a stable vertical band for boxes even on empty lines
            if line:
                t_l, t_t, t_r, t_b = draw.textbbox((x0, yy), line, font=font)
            else:
                # representative glyphs for ascent/descent
                t_l, t_t, t_r, t_b = draw.textbbox((x0, yy), "Ag", font=font)

            y_pad = int(1 * s)
            y_top = t_t - y_pad
            y_bot = t_b + y_pad

            # NEW: guarantee a minimum vertical size for all highlight boxes on this line.
            # This applies to:
            #  - whole-line boxes (back-compat)
            #  - per-token boxes
            #  - newline boxes
            y_top, y_bot = _ensure_min_box_height(y_top, y_bot, _min_box_h, render_H)

            # Draw highlight boxes
            if highlight_token_spans is None:
                # Back-compat: box whole line (empty line gets a minimal-width box)
                box_w = w(line) if line else max(w(" "), min_nl_box_px)
                bbox = (x0, y_top, x0 + box_w + int(2 * s), y_bot)
                # If block diffusion, force dotted border on highlighted region(s)
                draw_box(
                    draw,
                    bbox,
                    fill=box_fill,
                    outline=latest_outline if block_diffusion else None,
                    width=1 if block_diffusion else 0,
                    radius=int(corner_radius * s),
                )
            else:
                # Box per-token (no merging). A token may intersect multiple wrapped lines.
                if line:
                    for tok_idx, a, b in highlight_token_spans:
                        lo = max(a, line_start)
                        hi = min(b, line_end)
                        if hi <= lo:
                            continue

                        lo_rel = lo - line_start
                        hi_rel = hi - line_start

                        prefix = line[:lo_rel]
                        seg = line[lo_rel:hi_rel]

                        x_lo = x0 + w(prefix)
                        x_hi = x_lo + w(seg)

                        # NEW: tighten boxes / add inter-token spacing by shrinking box bounds.
                        # This makes adjacent highlighted tokens visually separable even when
                        # there is no literal whitespace between them.
                        x_lo_s = x_lo + _shrink_left
                        x_hi_s = x_hi - _shrink_right
                        if x_hi_s - x_lo_s < _min_token_box_w:
                            # Keep at least a tiny box; center it within the original span.
                            mid = (x_lo + x_hi) // 2
                            x_lo_s = mid - (_min_token_box_w // 2)
                            x_hi_s = x_lo_s + _min_token_box_w

                        lb = (x_lo_s, y_top, x_hi_s, y_bot)
                        # Non-block diffusion: gold with per-token alpha gradient.
                        if (not block_diffusion) and (tok_idx in gradient_tok_to_alpha):
                            apha = gradient_tok_to_alpha[tok_idx]
                            fillc = (box_fill[0], box_fill[1], box_fill[2], apha)
                        else:
                            fillc = box_fill
                        draw_box(
                            draw,
                            lb,
                            fill=fillc,
                            outline=None,   # no border on highlighted token boxes
                            width=0,
                            radius=int(corner_radius * s),
                        )

                # NEW: draw a box for the newline itself (works for "\n" and "\n\n" tokens)
                if hard_nl_after:
                    nl_idx = line_end  # index of '\n' in the original text
                    spans_ab = [(a, b) for (_ti, a, b) in highlight_token_spans]
                    if _span_covers(spans_ab, nl_idx) and _should_box_newline(
                        s_full, spans_ab, nl_idx
                    ):
                        # If a single highlighted token decodes to multiple consecutive newlines
                        # (e.g. "\n\n"), avoid boxing the earlier newline(s) at the end of the
                        # previous line; instead only box the newline that creates the empty line.
                        if _is_consecutive_newline_in_same_span(s_full, spans_ab, nl_idx):
                            pass
                        else:
                            # place right after the last visible char; empty line anchors at x0
                            x_lo = x0 + (w(line) if line else 0)
                            nl_w = max(w(" "), _min_nl_box_px)
                            x_hi = x_lo + nl_w

                            # NEW: apply same shrink logic to newline boxes so they match token boxes
                            x_lo_s = x_lo + _shrink_left
                            x_hi_s = x_hi - _shrink_right
                            if x_hi_s - x_lo_s < _min_token_box_w:
                                mid = (x_lo + x_hi) // 2
                                x_lo_s = mid - (_min_token_box_w // 2)
                                x_hi_s = x_lo_s + _min_token_box_w

                            lb = (x_lo_s, y_top, x_hi_s, y_bot)
                            # Match the gradient feel on newline boxes in non-block diffusion:
                            # treat newline as the *last* token in the window (most transparent).
                            if (not block_diffusion) and gradient_tok_to_alpha:
                                apha = min(gradient_tok_to_alpha.values())
                                fillc = (box_fill[0], box_fill[1], box_fill[2], apha)
                            else:
                                fillc = box_fill
                            draw_box(
                                draw,
                                lb,
                                fill=fillc,
                                outline=None,  # no border on newline highlight boxes either
                                width=0,
                                radius=int(corner_radius * s),
                            )

            # Draw text (note: invisible_spans are RAW; but our s_full is visible-only after removals.
            if line:
                # Build per-char highlight background spans so invisible text can match the
                # actual box color behind it (including alpha-composited gradients).
                highlight_bg_spans: Optional[List[Tuple[int, int, Tuple[int, int, int]]]] = None
                if highlight_token_spans is None:
                    # Whole-line highlight (back-compat): treat entire visible line as "boxed".
                    # If the line is empty, there is nothing to draw anyway.
                    highlight_bg_spans = [(line_start, line_end, box_fill)]
                else:
                    highlight_bg_spans = []
                    for tok_idx, a, b in highlight_token_spans:
                        # Determine the effective RGB behind text for this token span.
                        if (not block_diffusion) and (tok_idx in gradient_tok_to_alpha):
                            eff = _blend_over_white(box_fill, gradient_tok_to_alpha[tok_idx])
                        else:
                            # Opaque box fill
                            eff = box_fill
                        highlight_bg_spans.append((a, b, eff))
                draw_line_with_invisible_spans(
                    draw=draw,
                    x_start=x0,
                    y=yy,
                    line_text=line,
                    line_char_start=line_start,
                    invisible_spans=invisible_spans_vis,
                    highlight_bg_spans_global=highlight_bg_spans,
                    font=font,
                    fg=text_color,
                    bg_default=bg_color,
                )

            yy += lh + int(line_spacing * s)

        if downscale_output:
            im = im.resize((W, H), Image.LANCZOS)

        # Bake alpha into RGB so non-block diffusion gradient is visible.
        # (convert('RGB') alone discards alpha and destroys the gradient effect)
        im = _flatten_rgba_on_white(im)

        frames_rgb.append(im)

    final_text = prev_decoded_global.replace("✅", "✔").replace(mask_token, "")
    clean_frame = draw_text_only_frame(final_text)
    frames_rgb.append(clean_frame)

    fps = 1000.0 / float(frame_ms)
    last_frame_hold_s = 3.0

    save_frames_to_mp4(
        frames_rgb=frames_rgb,
        out_path=out_path,
        fps=fps,
        last_frame_hold_s=last_frame_hold_s,
    )


def gather_results(results, world_size):
    # Each GPU has local 'results' (any pickle-able object)
    if world_size == 1:
        return results
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)

    # gathered_results is now a list of lists (one per rank)
    all_results = []
    for partial in gathered_results:
        all_results.extend(partial)  # type: ignore

    return all_results


class LMEvalHarnessModel(LM):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        generated_samples_output_path: str,
        tokenizer: PreTrainedTokenizer,
        pretrained_model_revision: str | None = None,
        load_ema_weights: bool = False,
        ckpt_file: str = "best-rank0.pt",  # best-rank0.pt or latest-rank0.pt
        gen_kwargs: Any | None = None,
        accelerator: accelerate.Accelerator | None = None,
        throughput_run: bool = False,
        throughput_samples: int = 100,
        model_config_overrides: dict[str, Any] | None = None,
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Path to ckpt dir or HF model repo.
            generated_samples_output_path (str): Path to generated samples dir.
            tokenizer (str): Tokenizer name or path.
            pretrained_model_revision (Optional[str]): Revision (e.g., commit id)
                passed to `.from_pretrained` model instantiation.
            load_ema_weights (bool): Whether to load ema weights (for local ckpts).
            ckpt_file (str): Name of ckpt file (for local ckpts).
            gen_kwargs (dict): Generator kwargs.
                Ideally this should be passed via `lm_eval.evaluator.simple_evaluate`,
                however this method expects `gen_kwargs` as string with comma-separated
                arguments, which is not compatible in our hydra framework.
            throughput_run (bool): Whether to run the evaluation throughput.
            model_config_overrides (dict[str, Any]): Model config overrides.
        """
        super().__init__()
        self.generated_samples_output_path = generated_samples_output_path
        if not fsspec_exists(self.generated_samples_output_path):
            fsspec_mkdirs(self.generated_samples_output_path)
        self.accelerator = accelerator
        if self.accelerator is not None:
            device = self.accelerator.device
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._rank = 0
            self._world_size = 1
        self.device = torch.device(f"{device}")

        model_config_overrides = (
            {} if model_config_overrides is None else model_config_overrides
        )
        # Handle string input (JSON string) - parse it to dict
        if isinstance(model_config_overrides, str):
            import json
            model_config_overrides = json.loads(model_config_overrides)
        # Convert to plain dict if it's a DictConfig to ensure proper merging
        if isinstance(model_config_overrides, DictConfig):
            from omegaconf import OmegaConf
            model_config_overrides = OmegaConf.to_container(model_config_overrides, resolve=True)
        if fsspec_exists(os.path.join(pretrained_model_name_or_path, "config.yaml")):
            model = load_model_from_ckpt_dir_path(
                path_to_ckpt_dir=pretrained_model_name_or_path,
                load_ema_weights=load_ema_weights,
                ckpt_file=ckpt_file,
                device=self.device,
                **model_config_overrides,
            )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    token="HF_TOKEN_REMOVED",
                )
            except:  # Model not compatible with CausalLM
                model = AutoModelForMaskedLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    token="HF_TOKEN_REMOVED",
                )
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        self.gen_kwargs = gen_kwargs
        self.throughput_run = throughput_run
        self.throughput_samples = throughput_samples

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def generate_until(self, requests, **generation_kwargs):
        # TODO: Move this to utils file / perhaps use chat template
        def _tokenize(
            e,
            prefix_text: str | None = (
                f"Please reason step by step, and put your "
                + "final answer within $\\boxed{}$. "
            ),
        ):
            ctx = e["prefix"]
            if self.tokenizer.chat_template is not None:
                # Extract question part (before "Answer:" if it exists)
                if "\nAnswer:" in ctx:
                    question_part = ctx.split("\nAnswer:")[0]
                else:
                    question_part = ctx

                # Remove "Question: " prefix if present
                if "Question: " in question_part:
                    question_text = prefix_text + question_part.split("Question: ")[1]
                else:
                    question_text = question_part

                messages = [
                    {"role": "user", "content": question_text},
                ]
                ctx = self.apply_chat_template(messages)
            else:
                ctx = re.sub(
                    r"^####\s*(\d+)\s*$",
                    r"$\\boxed{\1}$" + self.tokenizer.eos_token,
                    ctx,
                    flags=re.MULTILINE,
                )
                ctx = ctx.replace("Question: ", prefix_text)
                ctx = ctx.replace("\nAnswer:", f"{self.tokenizer.eos_token}Answer:")
            prefix_tokens = self.tokenizer(ctx)["input_ids"]
            return {
                "prefix_text": ctx,
                "prefix": prefix_tokens,
                "target": e["target"],
            }

        from src.datasets.preprocessed_dataset import load_preprocessed_dataset

        ds = load_preprocessed_dataset(
            dataset_path="/share/kuleshov/ma2238/dllm-dev-new/dllm-dev/outputs/distillation/Qwen3-32B-AWQ/gsm8k_eval",
            inject_context_mask=True,
            tokenizer=self.tokenizer,
            token_to_split="<|im_start|>",
            split_offset=2
        )
        ds = ds.select(range(400, 401))
        
        self.throughput_samples = len(ds)
        tputs = []

        for i, elem in tqdm(
            enumerate(ds), desc="Generating", total=len(ds), disable=(self.rank != 0)
        ):
            full_text = self.tokenizer.decode(elem["input_ids"])
            target_text = full_text.split("<|im_start|>assistant\n")[1]
            prefix_text = full_text.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
            elem["prefix"] = prefix_text
            elem["target"] = target_text
            elem = _tokenize(elem)
            if (
                self.throughput_run
                and i >= self.throughput_samples
            ):
                tputs_path = (
                    f"{self.generated_samples_output_path}/throughput-rank{self.rank}"
                )
                with open(f"{tputs_path}.json", "w") as f:
                    json.dump(
                        {
                            "throughput_mean": np.mean(tputs),
                            "throughput_std": np.std(tputs),
                            "throughput_all": tputs,
                        },
                        f,  # type: ignore
                        indent=2,
                    )
                sys.exit(0)
            if self.rank == 0:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_event, end_event = None, None
            # add think tags to prefix

            # elem["prefix"] = elem["prefix"] + self.tokenizer("<think>\n\n</think>\n\n")["input_ids"]
            generation_outputs = self.model.generate(
                inputs=torch.tensor(elem["prefix"])[None, ...].to(self.device),
                disable_pbar=(self.rank != 0),
                tokenizer=self.tokenizer,  # Uncomment for debugging
                **self.gen_kwargs,
            )
            # cut off any text after boxed in all intermediate samples/windows
            # get last index of }
            # last_index_of_box = torch.where(generation_outputs.sequences[-1] == self.tokenizer.convert_tokens_to_ids("}"))[0][-1].item() + 1

            # input_offset = len(elem["prefix"])
            # new_intermediate_samples = []
            # for intm_sample in generation_outputs.intermediate_samples:
            #     new_intermediate_samples.append(intm_sample[:last_index_of_box - input_offset])
            # generation_outputs.intermediate_samples = new_intermediate_samples
            # new_intermediate_window_offsets = []
            # for intm_window_offset in generation_outputs.intermediate_window_offsets:
            #     new_intermediate_window_offsets.append(intm_window_offset[intm_window_offset < last_index_of_box - input_offset])
            # generation_outputs.intermediate_window_offsets = new_intermediate_window_offsets
            intm_samples = generation_outputs.intermediate_samples
            intm_window_offsets = generation_outputs.intermediate_window_offsets

            sample = generation_outputs.sequences
            import ipdb ; ipdb.set_trace()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_s = start_event.elapsed_time(end_event) / 1000
            if i >= 10:
                tputs.append((sample.numel() - len(elem["prefix"])) / elapsed_time_s)

            # setdlm tput: 53.74 +/- 0.70, 60.00 +/- 5.76
            # bd3lm tput: 36.35 +/- 0.13,  41.81 +/- 3.82
            # frame_ms = 300
            # frame_ms = 300
            # block_diffusion = True
            # strings_to_gif_oldstyle(
            #     tokenizer=self.tokenizer,
            #     tokens=intm_samples,
            #     highlight_token_indices=intm_window_offsets,
            #     out_path=f"strings_{i}_bd{block_diffusion}_v3.mp4",
            #     size=(1200, 400),
            #     scale=2,
            #     downscale_output=True,
            #     frame_ms=frame_ms,              # tweak to taste
            #     block_diffusion=block_diffusion,
            # )
            # print(f"Wrote strings_{i}.mp4")

            # all_lens = []
            # for j in intm_window_offsets:
            #     print(len(j))
            #     all_lens.append(len(j))
            # print(np.mean(all_lens))

            result = self.tokenizer.decode(sample[0, len(elem["prefix"]) :])
            if self.rank == 0:
                print("=" * 20)
                print("prefix: ", self.tokenizer.decode(elem["prefix"]), result)
                print("(Ground truth): ", elem["target"])
                print("=" * 20, end="\n\n")

        print(f"RANK {self.rank} completed!")
        return None


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    ipg = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator = accelerate.Accelerator(kwargs_handlers=[ipg])
    accelerator = accelerator if accelerator.num_processes > 1 else None
    set_seed(cfg.seed)
    model = hydra.utils.instantiate(cfg.task.model, accelerator=accelerator)
    results = hydra.utils.call(cfg.task, model=model)
    if results is not None and (
        accelerator is None or accelerator.local_process_index == 0
    ):
        samples = results.pop("samples")
        evaluation_tracker = EvaluationTracker(output_path=cfg.output_path)
        evaluation_tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(
                task_name=task_name, samples=samples[task_name]
            )
        print(make_table(results))
        metrics_f = f"{cfg.task.model.generated_samples_output_path}/metrics.txt"
        with open(metrics_f, "w") as f:
            f.write(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

if __name__ == "__main__":
    register_useful_resolvers()
    main()
