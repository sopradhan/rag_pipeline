"""vlm_demo.py
Small Vision-Language Model demo.

Behavior:
- If `transformers` + CLIP are available locally, it will use CLIP (model name from HF env or default) to produce a caption for an input image.
- Otherwise, falls back to a deterministic heuristic caption (dominant color + image size) so the demo runs offline.
- Saves an annotated output image `vlm_demo_output.png` and prints the caption.

Usage:
    python vlm_demo.py --image path/to/image.jpg

Notes:
- This file intentionally keeps heavy imports optional and provides a safe offline fallback.
- To use real CLIP, install `transformers` and `torch` and set HF model via env HF_CLIP_MODEL (e.g., 'openai/clip-vit-base-patch32' or 'openai/clip-vit-large-patch14').
"""
from __future__ import annotations
import os
import sys
import argparse
from typing import Optional
import sqlite3
import json
import datetime
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None


def _fallback_caption(image_path: str) -> str:
    """Produce a deterministic fallback caption using average color and dimensions."""
    try:
        from PIL import Image
    except Exception:
        return "Image caption: (PIL not available)"
    im = Image.open(image_path).convert('RGB')
    w, h = im.size
    # downsample and compute average color
    small = im.resize((32, 32))
    pixels = list(small.getdata())
    r = sum(p[0] for p in pixels) / len(pixels)
    g = sum(p[1] for p in pixels) / len(pixels)
    b = sum(p[2] for p in pixels) / len(pixels)
    dominant = max(('red', r), ('green', g), ('blue', b), key=lambda x: x[1])[0]
    caption = f"A {dominant}-tinted image of size {w}x{h}."
    return caption


def _annotate_image(image_path: str, caption: str, out_path: str = 'vlm_demo_output.png') -> None:
    if Image is None:
        print('Pillow not installed; skipping image annotation.')
        return
    im = Image.open(image_path).convert('RGBA')
    w, h = im.size
    # create overlay for text
    overlay = Image.new('RGBA', im.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    # choose font if available
    try:
        font = ImageFont.truetype('arial.ttf', size=max(12, w // 40))
    except Exception:
        font = ImageFont.load_default()
    # wrap caption to multiple lines
    max_width = max(40, w // 10)
    lines = []
    words = caption.split()
    cur = ''
    for tok in words:
        if len(cur) + 1 + len(tok) <= max_width:
            cur = (cur + ' ' + tok).strip()
        else:
            lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    # draw semi-transparent rectangle at bottom
    padding = 8
    # compatibility helper to get text size across Pillow versions
    def _text_size(draw_obj, txt, fnt):
        # preferred: ImageDraw.textbbox
        if hasattr(draw_obj, "textbbox"):
            bbox = draw_obj.textbbox((0, 0), txt, font=fnt)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        # older Pillow: ImageDraw.textsize
        if hasattr(draw_obj, "textsize"):
            return draw_obj.textsize(txt, font=fnt)
        # fallback: Font.getbbox or getmask
        if hasattr(fnt, "getbbox"):
            bbox = fnt.getbbox(txt)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        try:
            mask = fnt.getmask(txt)
            return mask.size
        except Exception:
            return (0, 0)

    line_heights = [_text_size(draw, l, font)[1] for l in lines]
    text_h = sum(h + 4 for h in line_heights)
    rect_h = text_h + 2 * padding
    draw.rectangle([(0, h - rect_h), (w, h)], fill=(0, 0, 0, 160))
    y = h - rect_h + padding
    for i, line in enumerate(lines):
        draw.text((padding, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_heights[i] + 4
    out = Image.alpha_composite(im, overlay)
    out.convert('RGB').save(out_path)
    print(f'Annotated image saved to {out_path}')


def _load_image_tensor(image_path: str) -> np.ndarray:
    """Load image as a numpy tensor (H,W,C) uint8."""
    try:
        from PIL import Image as PILImage
    except Exception:
        raise RuntimeError('Pillow is required to load image tensor')
    im = PILImage.open(image_path).convert('RGB')
    arr = np.array(im)
    return arr


def run_vlm_demo(image_path: str, model_name: Optional[str] = None) -> dict:
    """Use VLM (Vision Language Model) to generate a caption and detect objects.
    Returns a dict with keys: caption, objects, summary, annotated_path, model_used, tensor.
    Falls back to color/size analysis if VLM not available.
    """
    # Try to use VLM for open-ended captioning (no hardcoded categories)
    try:
        from transformers import pipeline
        # Create image-to-text pipeline (uses VLM like BLIP or similar)
        try:
            model_name = model_name or "Salesforce/blip-image-captioning-base"
            captioner = pipeline("image-to-text", model=model_name)
            # Try different prompt templates to get diverse views
            templates = [
                "A picture of",
                "This image shows",
                "The objects in this image are",
            ]
            captions = []
            for template in templates:
                try:
                    result = captioner(image_path, max_new_tokens=77, prompt=template)
                    if isinstance(result, list) and result:
                        if 'generated_text' in result[0]:
                            captions.append(result[0]['generated_text'])
                        elif isinstance(result[0], str):
                            captions.append(result[0])
                except Exception:
                    continue

            if not captions:
                raise RuntimeError("VLM generated no valid captions")

            caption = max(captions, key=len)
            objects = []
            for cap in captions:
                objs = _extract_objects_from_caption(cap)
                for obj in objs:
                    if obj not in objects:
                        objects.append(obj)

            summary = f"Primary description: {caption}\n"
            if len(captions) > 1:
                summary += f"Additional details: {'; '.join(c for c in captions if c != caption)}\n"
            if objects:
                summary += f"Detected objects/elements: {', '.join(objects)}"

            annotated_path = 'vlm_demo_output.png'
            _annotate_image(image_path, caption, out_path=annotated_path)
            tensor = _load_image_tensor(image_path)
            return {
                'caption': caption,
                'objects': objects,
                'summary': summary,
                'annotated_path': annotated_path,
                'model_used': model_name,
                'tensor': tensor,
            }
        except Exception as e:
            print(f"VLM error: {e}")
            # fall through to fallback
            pass
    except Exception:
        # transformers not available — fallback
        pass

    # Fallback: use basic image properties
    caption = _fallback_caption(image_path)
    annotated_path = 'vlm_demo_output.png'
    _annotate_image(image_path, caption, out_path=annotated_path)
    objects = _extract_objects_from_caption(caption)
    summary = _summarize_caption(caption, objects)
    tensor = _load_image_tensor(image_path)
    return {
        'caption': caption,
        'objects': objects,
        'summary': summary,
        'annotated_path': annotated_path,
        'model_used': None,
        'tensor': tensor,
    }


def _extract_objects_from_caption(caption: str):
    """Very light-weight heuristic to extract object phrases from a caption.
    Splits on common delimiters and returns nouns/phrases of interest.
    This is intentionally simple so it works offline.
    """
    # naive: split by commas and 'with' / 'showing'
    parts = [p.strip() for p in caption.replace('showing', ',').replace('with', ',').split(',') if p.strip()]
    # keep parts that are short and likely objects
    objs = [p for p in parts if 1 <= len(p.split()) <= 4]
    return objs[:5]


def _summarize_caption(caption: str, objects: list[str]):
    """Produce a concise 1-2 line summary from a caption and object list (deterministic).
    If a real LLM is available, this could be replaced with a call to an LLM for richer summaries.
    """
    objs = ', '.join(objects) if objects else 'no obvious distinct objects'
    summary = f"This image appears to contain: {objs}.\nShort description: {caption}"
    return summary


def _serialize_tensor_blob(arr: np.ndarray) -> tuple[bytes, dict]:
    """Serialize numpy array to bytes and return (blob, meta) where meta contains shape and dtype."""
    meta = {'shape': arr.shape, 'dtype': str(arr.dtype)}
    blob = arr.tobytes()
    return blob, meta


def save_to_sqlite(db_path: str, image_path: str, vlm_record: dict) -> int:
    """Save the VLM record to a SQLite database. Stores tensor as BLOB and metadata as JSON.

    Returns the inserted row id.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        image_path TEXT,
        annotated_path TEXT,
        caption TEXT,
        summary TEXT,
        objects_json TEXT,
        tensor_blob BLOB,
        tensor_meta_json TEXT,
        width INTEGER,
        height INTEGER,
        mode TEXT,
        model_used TEXT,
        created_at TEXT
    )
    ''')
    caption = vlm_record.get('caption')
    summary = vlm_record.get('summary')
    objects_json = json.dumps(vlm_record.get('objects', []))
    tensor = vlm_record.get('tensor')
    blob, meta = _serialize_tensor_blob(tensor)
    tensor_meta_json = json.dumps(meta)
    from PIL import Image as PILImage
    im = PILImage.open(image_path)
    w, h = im.size
    mode = im.mode
    model_used = vlm_record.get('model_used')
    created_at = datetime.datetime.utcnow().isoformat()
    cur.execute(
        'INSERT INTO images (image_path, annotated_path, caption, summary, objects_json, tensor_blob, tensor_meta_json, width, height, mode, model_used, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        (image_path, vlm_record.get('annotated_path'), caption, summary, objects_json, sqlite3.Binary(blob), tensor_meta_json, w, h, mode, model_used, created_at)
    )
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    print(f'Saved record id={rowid} to {db_path}')
    return rowid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple VLM demo (CLIP if available, else fallback)')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', default=None, help='Optional model name to use (VLM)')
    parser.add_argument('--db', '-d', default=None, help='Optional SQLite DB path to save results')
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        print('Image not found:', args.image)
        sys.exit(1)
    record = run_vlm_demo(args.image, model_name=args.model)
    # attach image_path into record for saving
    if args.db:
        try:
            save_to_sqlite(args.db, args.image, record)
        except Exception as e:
            print('Failed to save to DB:', e)
    else:
        print('No DB path supplied; skipping save_to_sqlite.')
