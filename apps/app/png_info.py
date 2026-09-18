from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PngInfo:
    width: int
    height: int
    bit_depth: int
    color_type: int
    file_size: int
    sha256: str

    @property
    def color_mode(self) -> str:
        return {2: "RGB", 6: "RGBA"}.get(self.color_type, f"TYPE_{self.color_type}")


def inspect_png(path: str | Path, calculate_hash: bool = True) -> PngInfo:
    file_path = Path(path)
    try:
        file_size = file_path.stat().st_size
        with file_path.open("rb") as stream:
            signature = stream.read(8)
            if signature != PNG_SIGNATURE:
                raise PngValidationError("not_png")

            chunk_length_raw = stream.read(4)
            chunk_type = stream.read(4)
            if len(chunk_length_raw) != 4 or chunk_type != b"IHDR":
                raise PngValidationError("missing_ihdr")

            chunk_length = struct.unpack(">I", chunk_length_raw)[0]
            if chunk_length != 13:
                raise PngValidationError("invalid_ihdr_length")

            ihdr = stream.read(13)
            if len(ihdr) != 13:
                raise PngValidationError("truncated_ihdr")

            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", ihdr
            )
            if compression != 0 or filtering != 0 or interlace not in (0, 1):
                raise PngValidationError("unsupported_png_header")
    except OSError as exc:
        raise PngValidationError(f"io_error:{exc}") from exc

    sha256 = ""
    if calculate_hash:
        digest = hashlib.sha256()
        try:
            with file_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise PngValidationError(f"io_error:{exc}") from exc
        sha256 = digest.hexdigest()

    return PngInfo(
        width=width,
        height=height,
        bit_depth=bit_depth,
        color_type=color_type,
        file_size=file_size,
        sha256=sha256,
    )


def validate_brush_png(path: str | Path) -> PngInfo:
    info = inspect_png(path)
    if info.width != 512 or info.height != 512:
        raise PngValidationError(f"invalid_size:{info.width}x{info.height}")
    if info.color_type not in (2, 6):
        raise PngValidationError(f"invalid_color_type:{info.color_type}")
    if info.bit_depth != 8:
        raise PngValidationError(f"invalid_bit_depth:{info.bit_depth}")
    return info
