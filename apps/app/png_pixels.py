from __future__ import annotations

import binascii
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngPixelError(ValueError):
    pass


@dataclass(frozen=True)
class PixelImage:
    width: int
    height: int
    channels: int
    pixels: bytes

    @property
    def stride(self) -> int:
        return self.width * self.channels


def sample_average_rgb(image: PixelImage) -> tuple[int, int, int]:
    """Return an alpha-weighted representative color from at most ~64x64 samples."""
    step_x = max(1, image.width // 64)
    step_y = max(1, image.height // 64)
    red = green = blue = alpha_total = 0
    samples = 0
    channels = image.channels
    for y in range(0, image.height, step_y):
        for x in range(0, image.width, step_x):
            index = (y * image.width + x) * channels
            alpha = image.pixels[index + 3] if channels == 4 else 255
            red += image.pixels[index] * alpha
            green += image.pixels[index + 1] * alpha
            blue += image.pixels[index + 2] * alpha
            alpha_total += alpha
            samples += 1
    if alpha_total == 0 or samples == 0:
        return (46, 56, 66)
    return (
        round(red / alpha_total),
        round(green / alpha_total),
        round(blue / alpha_total),
    )


def read_png_pixels(path: str | Path) -> PixelImage:
    source_path = Path(path)
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise PngPixelError(f"Cannot read PNG: {exc}") from exc
    if len(payload) < 8 or payload[:8] != PNG_SIGNATURE:
        raise PngPixelError("Invalid PNG signature")

    offset = 8
    width = height = bit_depth = color_type = interlace = None
    compressed_parts: list[bytes] = []
    seen_iend = False
    while offset + 12 <= len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        offset += 4
        chunk_type = payload[offset : offset + 4]
        offset += 4
        if offset + length + 4 > len(payload):
            raise PngPixelError("PNG chunk is truncated")
        chunk_data = payload[offset : offset + length]
        offset += length
        expected_crc = struct.unpack_from(">I", payload, offset)[0]
        offset += 4
        actual_crc = binascii.crc32(chunk_type)
        actual_crc = binascii.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise PngPixelError(f"PNG chunk CRC mismatch: {chunk_type!r}")
        if chunk_type == b"IHDR":
            if length != 13:
                raise PngPixelError("Invalid IHDR size")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if compression != 0 or filtering != 0:
                raise PngPixelError("Unsupported PNG compression or filter method")
        elif chunk_type == b"IDAT":
            compressed_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            seen_iend = True
            break

    if not seen_iend or width is None or height is None:
        raise PngPixelError("PNG is missing IHDR or IEND")
    if bit_depth != 8 or color_type not in (2, 6):
        raise PngPixelError("Only 8-bit RGB and RGBA PNG files are supported")
    if interlace not in (0, 1):
        raise PngPixelError(f"Unsupported PNG interlace method {interlace}")
    channels = 3 if color_type == 2 else 4
    try:
        filtered = zlib.decompress(b"".join(compressed_parts))
    except zlib.error as exc:
        raise PngPixelError(f"PNG decompression failed: {exc}") from exc

    if interlace == 0:
        row_size = width * channels
        expected_size = height * (row_size + 1)
        if len(filtered) != expected_size:
            raise PngPixelError(
                f"PNG scanline size mismatch: expected {expected_size}, received {len(filtered)}"
            )
        output = bytearray(width * height * channels)
        previous = bytearray(row_size)
        source_offset = 0
        output_offset = 0
        for _row in range(height):
            filter_type = filtered[source_offset]
            source_offset += 1
            scanline = bytearray(filtered[source_offset : source_offset + row_size])
            source_offset += row_size
            _undo_filter(scanline, previous, channels, filter_type)
            output[output_offset : output_offset + row_size] = scanline
            output_offset += row_size
            previous = scanline
        return PixelImage(width, height, channels, bytes(output))

    return _decode_adam7(filtered, width, height, channels)


def _decode_adam7(filtered: bytes, width: int, height: int, channels: int) -> PixelImage:
    passes = (
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    )
    output = bytearray(width * height * channels)
    source_offset = 0
    for start_x, start_y, step_x, step_y in passes:
        pass_width = 0 if width <= start_x else (width - start_x + step_x - 1) // step_x
        pass_height = 0 if height <= start_y else (height - start_y + step_y - 1) // step_y
        if pass_width == 0 or pass_height == 0:
            continue
        row_size = pass_width * channels
        previous = bytearray(row_size)
        for pass_y in range(pass_height):
            if source_offset >= len(filtered):
                raise PngPixelError("Adam7 scanline data is truncated")
            filter_type = filtered[source_offset]
            source_offset += 1
            if source_offset + row_size > len(filtered):
                raise PngPixelError("Adam7 scanline payload is truncated")
            scanline = bytearray(filtered[source_offset : source_offset + row_size])
            source_offset += row_size
            _undo_filter(scanline, previous, channels, filter_type)
            target_y = start_y + pass_y * step_y
            for pass_x in range(pass_width):
                target_x = start_x + pass_x * step_x
                source_index = pass_x * channels
                target_index = (target_y * width + target_x) * channels
                output[target_index : target_index + channels] = scanline[
                    source_index : source_index + channels
                ]
            previous = scanline
    if source_offset != len(filtered):
        raise PngPixelError(
            f"Adam7 scanline size mismatch: consumed {source_offset}, received {len(filtered)}"
        )
    return PixelImage(width, height, channels, bytes(output))


def _undo_filter(scanline: bytearray, previous: bytearray, bytes_per_pixel: int, filter_type: int) -> None:
    if filter_type == 0:
        return
    for index in range(len(scanline)):
        left = scanline[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        up = previous[index]
        up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        if filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = _paeth(left, up, up_left)
        else:
            raise PngPixelError(f"Unsupported PNG filter type {filter_type}")
        scanline[index] = (scanline[index] + predictor) & 0xFF


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    distance_left = abs(estimate - left)
    distance_up = abs(estimate - up)
    distance_up_left = abs(estimate - up_left)
    if distance_left <= distance_up and distance_left <= distance_up_left:
        return left
    if distance_up <= distance_up_left:
        return up
    return up_left


def resize_nearest(image: PixelImage, width: int, height: int) -> PixelImage:
    if width < 1 or height < 1:
        raise ValueError("Target dimensions must be positive")
    if image.width == width and image.height == height:
        return image
    channels = image.channels
    output = bytearray(width * height * channels)
    for target_y in range(height):
        source_y = min(image.height - 1, (target_y * image.height) // height)
        for target_x in range(width):
            source_x = min(image.width - 1, (target_x * image.width) // width)
            source_index = (source_y * image.width + source_x) * channels
            target_index = (target_y * width + target_x) * channels
            output[target_index : target_index + channels] = image.pixels[
                source_index : source_index + channels
            ]
    return PixelImage(width, height, channels, bytes(output))


def add_edge_padding(image: PixelImage, padding: int = 4) -> PixelImage:
    if padding < 0:
        raise ValueError("Padding cannot be negative")
    if padding == 0:
        return image
    width = image.width + padding * 2
    height = image.height + padding * 2
    channels = image.channels
    output = bytearray(width * height * channels)
    for target_y in range(height):
        source_y = min(image.height - 1, max(0, target_y - padding))
        for target_x in range(width):
            source_x = min(image.width - 1, max(0, target_x - padding))
            source_index = (source_y * image.width + source_x) * channels
            target_index = (target_y * width + target_x) * channels
            output[target_index : target_index + channels] = image.pixels[
                source_index : source_index + channels
            ]
    return PixelImage(width, height, channels, bytes(output))


def encode_png(image: PixelImage) -> bytes:
    if image.channels not in (3, 4):
        raise PngPixelError("Only RGB and RGBA output is supported")
    color_type = 2 if image.channels == 3 else 6
    row_size = image.width * image.channels
    raw = bytearray()
    for row in range(image.height):
        raw.append(0)
        start = row * row_size
        raw.extend(image.pixels[start : start + row_size])
    header = struct.pack(">IIBBBBB", image.width, image.height, 8, color_type, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + _chunk(b"IEND", b"")


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def atomic_write_png(path: str | Path, image: PixelImage, *, durable: bool = True) -> None:
    """Write a PNG through a temporary file and an atomic rename.

    ``durable=False`` skips the fsync. Use it only for caches that are
    regenerated from authoritative data: an fsync costs tens of milliseconds on
    Windows, which dominates the write of a small derived file, and losing such a
    file to a crash only means rebuilding it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encode_png(image))
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
