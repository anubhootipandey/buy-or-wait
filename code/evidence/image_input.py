"""
Stage 4d image input handling.

Reads an evidence image off disk into the exact three things the vision
resolver needs and nothing else:

  * `data`      - the raw bytes to hand to the model,
  * `mime_type` - the image's ACTUAL type, sniffed from its magic bytes
                  rather than assumed from the filename. The dataset
                  currently ships PNGs, but a real evidence pipeline
                  receives whatever the user uploaded, and sending a JPEG
                  labelled `image/png` is a silent, hard-to-debug failure
                  mode at the provider.
  * `sha256`    - a content hash, used as the cache key component so two
                  different images can never collide and an edited image
                  can never be served a stale cached fact.

This module does no I/O beyond reading the file, calls no model, and has
no dependency on the rest of the evidence package - it is deliberately
importable on its own.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# MIME types the vision path accepts. This is the intersection of "types
# Gemini accepts natively" and "types we are willing to claim we handle".
# Anything else is refused up front rather than sent and hoped for.
SUPPORTED_IMAGE_MIME_TYPES = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
)

# Magic-byte signatures, checked in order. Each entry is
# (offset, signature_bytes, mime_type).
_MAGIC_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"\x89PNG\r\n\x1a\n", "image/png"),
    (0, b"\xff\xd8\xff", "image/jpeg"),
    (0, b"GIF87a", "image/gif"),
    (0, b"GIF89a", "image/gif"),
)

# Extension fallback, used ONLY when magic-byte sniffing is inconclusive.
_EXTENSION_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
}

# Refuse absurdly large uploads before spending an API call on them.
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class UnsupportedImageError(Exception):
    """Raised when an image cannot be safely prepared for the model.

    Callers turn this into an `UnresolvedEvidence` - it is never allowed
    to become a fact, and never crashes the pipeline.
    """


@dataclass(frozen=True)
class ImagePayload:
    """One evidence image, ready to send."""

    data: bytes
    mime_type: str
    sha256: str
    byte_size: int


def sniff_mime_type(data: bytes, path: Path | str | None = None) -> str | None:
    """Return the image's MIME type from its magic bytes, falling back to
    its file extension. Returns None when neither is conclusive.

    Magic bytes win over the extension: a file named `.png` whose bytes
    are a JPEG is a JPEG, and mislabelling it at the provider is exactly
    the bug this function exists to prevent.
    """
    for offset, signature, mime_type in _MAGIC_SIGNATURES:
        if data[offset : offset + len(signature)] == signature:
            return mime_type

    # RIFF-container WebP: "RIFF" .... "WEBP".
    if data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    # ISO-BMFF branded HEIC/HEIF: box type "ftyp" at offset 4.
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"heim", b"heis"):
            return "image/heic"
        if brand in (b"mif1", b"msf1"):
            return "image/heif"

    if path is not None:
        return _EXTENSION_MIME_TYPES.get(Path(path).suffix.lower())
    return None


def load_image(path: Path | str) -> ImagePayload:
    """Read `path` into an `ImagePayload`.

    Raises `UnsupportedImageError` for a missing file, an empty file, an
    oversized file, or a file whose type we cannot establish/accept. It
    never guesses a MIME type it is not confident about.
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except FileNotFoundError as exc:
        raise UnsupportedImageError(f"image file not found: {p}") from exc
    except OSError as exc:
        raise UnsupportedImageError(f"image file could not be read: {p} ({exc})") from exc

    if not data:
        raise UnsupportedImageError(f"image file is empty: {p}")
    if len(data) > MAX_IMAGE_BYTES:
        raise UnsupportedImageError(
            f"image file is too large: {len(data)} bytes exceeds the "
            f"{MAX_IMAGE_BYTES}-byte limit ({p})"
        )

    mime_type = sniff_mime_type(data, p)
    if mime_type is None:
        raise UnsupportedImageError(
            f"could not determine image type from content or extension: {p}"
        )
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise UnsupportedImageError(f"unsupported image type {mime_type!r}: {p}")

    return ImagePayload(
        data=data,
        mime_type=mime_type,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
    )


__all__ = [
    "ImagePayload",
    "UnsupportedImageError",
    "SUPPORTED_IMAGE_MIME_TYPES",
    "MAX_IMAGE_BYTES",
    "load_image",
    "sniff_mime_type",
]
