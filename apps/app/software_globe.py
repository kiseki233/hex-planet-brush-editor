from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping

from .brush_catalog import BrushRecord
from .distant_lod import DistantLodCache, EMPTY_RGB, MISSING_RGB
from .map_store import BrushTableEntry
from .png_pixels import PixelImage


@dataclass(frozen=True)
class ProjectedCellPolygon:
    cell_id: int
    points: tuple[float, ...]
    depth: float
    fill: str
    edge: str


def rotate_point(
    point: tuple[float, float, float], yaw: float, pitch: float
) -> tuple[float, float, float]:
    x, y, z = point
    cosine_yaw = math.cos(yaw)
    sine_yaw = math.sin(yaw)
    x, z = x * cosine_yaw + z * sine_yaw, -x * sine_yaw + z * cosine_yaw
    cosine_pitch = math.cos(pitch)
    sine_pitch = math.sin(pitch)
    return x, y * cosine_pitch - z * sine_pitch, y * sine_pitch + z * cosine_pitch


def draw_shaded_sphere(
    canvas,
    center_x: float,
    center_y: float,
    radius: float,
    *,
    layers: int = 56,
    tags: tuple[str, ...] = (),
) -> None:
    """Draw a continuous shaded sphere using nested circles.

    Tk does not expose a hardware depth buffer or an alpha-gradient primitive.  A
    bounded stack of shifted circles gives a continuous sphere silhouette without
    pretending that far-view aggregate nodes are editable hexagons.
    """
    if radius <= 0.0:
        return
    dark = (22, 32, 39)
    mid = (55, 78, 91)
    light = (128, 157, 170)
    canvas.create_oval(
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
        fill=_hex(dark),
        outline="",
        tags=tags,
    )
    count = max(12, int(layers))
    for index in range(count):
        progress = (index + 1) / count
        scale = 1.0 - progress * 0.82
        ring_radius = radius * scale
        shift_x = -radius * 0.20 * progress
        shift_y = -radius * 0.23 * progress
        if progress < 0.62:
            mix = progress / 0.62
            color = _mix(dark, mid, mix)
        else:
            mix = (progress - 0.62) / 0.38
            color = _mix(mid, light, mix)
        canvas.create_oval(
            center_x + shift_x - ring_radius,
            center_y + shift_y - ring_radius,
            center_x + shift_x + ring_radius,
            center_y + shift_y + ring_radius,
            fill=_hex(color),
            outline="",
            tags=tags,
        )
    canvas.create_oval(
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
        fill="",
        outline="#c8d7de",
        width=2,
        tags=tags,
    )




def render_textured_globe_view_ppm(
    texture: PixelImage,
    yaw: float,
    pitch: float,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius: float,
    *,
    block_size: int = 3,
    background: tuple[int, int, int] = (20, 25, 31),
) -> bytes:
    """Render the visible part of a textured orthographic globe into a PPM image."""
    width = max(1, int(width))
    height = max(1, int(height))
    radius = max(1.0, float(radius))
    block_size = max(1, int(block_size))
    if texture.channels not in (3, 4):
        raise ValueError("Software globe texture must be RGB or RGBA")
    pixels = bytearray(bytes(background) * (width * height))
    cosine_yaw = math.cos(yaw)
    sine_yaw = math.sin(yaw)
    cosine_pitch = math.cos(pitch)
    sine_pitch = math.sin(pitch)
    source = texture.pixels
    channels = texture.channels
    texture_width = texture.width
    texture_height = texture.height
    light = _normalize3((-0.38, 0.48, 0.79))
    sample_key = (
        width,
        height,
        round(center_x, 2),
        round(center_y, 2),
        round(radius, 2),
        block_size,
    )
    for x0, y0, camera_x, camera_y, camera_z in _globe_view_samples(*sample_key):
        intermediate_y = camera_y * cosine_pitch + camera_z * sine_pitch
        intermediate_z = -camera_y * sine_pitch + camera_z * cosine_pitch
        world_x = camera_x * cosine_yaw - intermediate_z * sine_yaw
        world_y = intermediate_y
        world_z = camera_x * sine_yaw + intermediate_z * cosine_yaw
        longitude = math.atan2(world_z, world_x)
        latitude = math.asin(max(-1.0, min(1.0, world_y)))
        source_x = int((longitude + math.pi) / (2.0 * math.pi) * texture_width) % texture_width
        source_y = min(
            texture_height - 1,
            max(0, int((math.pi / 2.0 - latitude) / math.pi * texture_height)),
        )
        source_offset = (source_y * texture_width + source_x) * channels
        red = source[source_offset]
        green = source[source_offset + 1]
        blue = source[source_offset + 2]
        if channels == 4:
            alpha = source[source_offset + 3] / 255.0
            red = round(red * alpha + background[0] * (1.0 - alpha))
            green = round(green * alpha + background[1] * (1.0 - alpha))
            blue = round(blue * alpha + background[2] * (1.0 - alpha))
        diffuse = max(0.0, camera_x * light[0] + camera_y * light[1] + camera_z * light[2])
        limb = max(0.0, min(1.0, camera_z))
        factor = 0.48 + diffuse * 0.46 + limb * 0.08
        color = (
            max(0, min(255, round(red * factor))),
            max(0, min(255, round(green * factor))),
            max(0, min(255, round(blue * factor))),
        )
        row_payload = bytes(color) * min(block_size, width - x0)
        for target_y in range(y0, min(height, y0 + block_size)):
            target_offset = (target_y * width + x0) * 3
            pixels[target_offset : target_offset + len(row_payload)] = row_payload
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + bytes(pixels)


@lru_cache(maxsize=12)
def _globe_view_samples(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius: float,
    block_size: int,
) -> tuple[tuple[int, int, float, float, float], ...]:
    result: list[tuple[int, int, float, float, float]] = []
    for y0 in range(0, height, block_size):
        sample_y = min(height - 0.5, y0 + block_size * 0.5)
        camera_y = -(sample_y - center_y) / radius
        if camera_y < -1.0 or camera_y > 1.0:
            continue
        for x0 in range(0, width, block_size):
            sample_x = min(width - 0.5, x0 + block_size * 0.5)
            camera_x = (sample_x - center_x) / radius
            radius_squared = camera_x * camera_x + camera_y * camera_y
            if radius_squared > 1.0:
                continue
            camera_z = math.sqrt(max(0.0, 1.0 - radius_squared))
            result.append((x0, y0, camera_x, camera_y, camera_z))
    return tuple(result)

def render_textured_sphere_ppm(
    texture: PixelImage,
    yaw: float,
    pitch: float,
    diameter: int,
    *,
    block_size: int = 2,
    background: tuple[int, int, int] = (20, 25, 31),
) -> bytes:
    """Render an equirectangular surface texture onto an orthographic sphere.

    The returned binary PPM can be passed directly to Tk PhotoImage. Rendering is
    block based so interaction can use a coarser block size without changing the
    displayed sphere diameter.
    """
    diameter = max(8, int(diameter))
    block_size = max(1, int(block_size))
    if texture.channels not in (3, 4):
        raise ValueError("Software globe texture must be RGB or RGBA")
    pixels = bytearray(bytes(background) * (diameter * diameter))
    cosine_yaw = math.cos(yaw)
    sine_yaw = math.sin(yaw)
    cosine_pitch = math.cos(pitch)
    sine_pitch = math.sin(pitch)
    source = texture.pixels
    channels = texture.channels
    width = texture.width
    height = texture.height
    light = _normalize3((-0.38, 0.48, 0.79))
    for x0, y0, camera_x, camera_y, camera_z in _sphere_samples(diameter, block_size):
        # Invert the forward pitch, then invert the forward yaw.
        intermediate_y = camera_y * cosine_pitch + camera_z * sine_pitch
        intermediate_z = -camera_y * sine_pitch + camera_z * cosine_pitch
        world_x = camera_x * cosine_yaw - intermediate_z * sine_yaw
        world_y = intermediate_y
        world_z = camera_x * sine_yaw + intermediate_z * cosine_yaw
        longitude = math.atan2(world_z, world_x)
        latitude = math.asin(max(-1.0, min(1.0, world_y)))
        source_x = int((longitude + math.pi) / (2.0 * math.pi) * width) % width
        source_y = min(
            height - 1,
            max(0, int((math.pi / 2.0 - latitude) / math.pi * height)),
        )
        source_offset = (source_y * width + source_x) * channels
        red = source[source_offset]
        green = source[source_offset + 1]
        blue = source[source_offset + 2]
        if channels == 4:
            alpha = source[source_offset + 3] / 255.0
            red = round(red * alpha + background[0] * (1.0 - alpha))
            green = round(green * alpha + background[1] * (1.0 - alpha))
            blue = round(blue * alpha + background[2] * (1.0 - alpha))
        diffuse = max(0.0, camera_x * light[0] + camera_y * light[1] + camera_z * light[2])
        limb = max(0.0, min(1.0, camera_z))
        factor = 0.48 + diffuse * 0.46 + limb * 0.08
        color = (
            max(0, min(255, round(red * factor))),
            max(0, min(255, round(green * factor))),
            max(0, min(255, round(blue * factor))),
        )
        row_payload = bytes(color) * min(block_size, diameter - x0)
        for target_y in range(y0, min(diameter, y0 + block_size)):
            target_offset = (target_y * diameter + x0) * 3
            pixels[target_offset : target_offset + len(row_payload)] = row_payload
    header = f"P6\n{diameter} {diameter}\n255\n".encode("ascii")
    return header + bytes(pixels)


@lru_cache(maxsize=12)
def _sphere_samples(
    diameter: int, block_size: int
) -> tuple[tuple[int, int, float, float, float], ...]:
    radius = diameter / 2.0
    if radius <= 0.0:
        return ()
    result: list[tuple[int, int, float, float, float]] = []
    for y0 in range(0, diameter, block_size):
        center_y = min(diameter - 0.5, y0 + block_size * 0.5)
        camera_y = -(center_y - radius) / radius
        for x0 in range(0, diameter, block_size):
            center_x = min(diameter - 0.5, x0 + block_size * 0.5)
            camera_x = (center_x - radius) / radius
            radius_squared = camera_x * camera_x + camera_y * camera_y
            if radius_squared > 1.0:
                continue
            camera_z = math.sqrt(max(0.0, 1.0 - radius_squared))
            result.append((x0, y0, camera_x, camera_y, camera_z))
    return tuple(result)


def _normalize3(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)
    if length <= 1e-15:
        return (0.0, 0.0, 1.0)
    return value[0] / length, value[1] / length, value[2] / length

def project_cell_polygon(
    topology,
    cell_id: int,
    yaw: float,
    pitch: float,
    center_x: float,
    center_y: float,
    radius: float,
    width: int,
    height: int,
    *,
    margin: float = 8.0,
) -> tuple[tuple[float, ...], float] | None:
    center = rotate_point(topology.cell_center(cell_id), yaw, pitch)
    if center[2] <= 0.0:
        return None
    projected: list[float] = []
    corners = topology.cell_corners(cell_id)
    if len(corners) != 6:
        return None
    for corner in corners:
        x, y, z = rotate_point(corner, yaw, pitch)
        # Cells crossing the limb are intentionally left to the continuous sphere
        # underlay.  This avoids drawing a polygon through the back hemisphere.
        if z <= 0.0:
            return None
        projected.extend((center_x + x * radius, center_y - y * radius))
    xs = projected[0::2]
    ys = projected[1::2]
    if max(xs) < -margin or min(xs) > width + margin:
        return None
    if max(ys) < -margin or min(ys) > height + margin:
        return None
    return tuple(projected), center[2]


def build_projected_cells(
    topology,
    layout,
    store,
    session,
    records_by_uid: Mapping[str, BrushRecord],
    brush_root,
    chunk_ids: Iterable[int],
    yaw: float,
    pitch: float,
    center_x: float,
    center_y: float,
    radius: float,
    width: int,
    height: int,
    *,
    maximum_cells: int = 8000,
) -> tuple[ProjectedCellPolygon, ...]:
    color_cache = DistantLodCache(brush_root)
    projected_cells: list[ProjectedCellPolygon] = []
    for chunk_id in chunk_ids:
        cell_ids = layout.chunk_cell_ids(int(chunk_id))
        values = session.loaded_chunks.get(int(chunk_id))
        if values is None:
            values = store.read_chunk_values(session, int(chunk_id))
        for local_index, cell_id in enumerate(cell_ids):
            if cell_id < 12:
                continue
            projected = project_cell_polygon(
                topology,
                cell_id,
                yaw,
                pitch,
                center_x,
                center_y,
                radius,
                width,
                height,
            )
            if projected is None:
                continue
            points, depth = projected
            value = values[local_index]
            local_id = value & 0x0FFF
            color = EMPTY_RGB
            if local_id:
                entry: BrushTableEntry | None = session.brush_entries.get(local_id)
                record = None if entry is None else records_by_uid.get(entry.brush_uid)
                color = color_cache.representative_color(record) if record is not None else MISSING_RGB
            shaded = shade_color(color, depth)
            edge = shade_color(shaded, 0.48)
            projected_cells.append(
                ProjectedCellPolygon(cell_id, points, depth, _hex(shaded), _hex(edge))
            )
            if len(projected_cells) > maximum_cells:
                return ()
    projected_cells.sort(key=lambda item: item.depth)
    return tuple(projected_cells)


def shade_color(color: tuple[int, int, int], depth: float) -> tuple[int, int, int]:
    depth = max(0.0, min(1.0, float(depth)))
    factor = 0.48 + depth * 0.62
    return tuple(max(0, min(255, round(channel * factor))) for channel in color)


def point_in_polygon(x: float, y: float, points: tuple[float, ...]) -> bool:
    inside = False
    count = len(points) // 2
    previous = count - 1
    for current in range(count):
        current_x = points[current * 2]
        current_y = points[current * 2 + 1]
        previous_x = points[previous * 2]
        previous_y = points[previous * 2 + 1]
        crosses = ((current_y > y) != (previous_y > y)) and (
            x
            < (previous_x - current_x) * (y - current_y)
            / ((previous_y - current_y) or 1e-15)
            + current_x
        )
        if crosses:
            inside = not inside
        previous = current
    return inside


def average_edge_pixels(points: tuple[float, ...]) -> float:
    count = len(points) // 2
    if count < 2:
        return 0.0
    total = 0.0
    for index in range(count):
        next_index = (index + 1) % count
        dx = points[index * 2] - points[next_index * 2]
        dy = points[index * 2 + 1] - points[next_index * 2 + 1]
        total += math.hypot(dx, dy)
    return total / count


def _mix(
    first: tuple[int, int, int], second: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(
        round(first[index] * (1.0 - amount) + second[index] * amount)
        for index in range(3)
    )


def _hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
