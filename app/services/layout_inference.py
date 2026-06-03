import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Placement:
    model_id: str
    position: tuple[float, float, float]
    rotation_y: float = 0.0
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    dimensions: dict = field(default_factory=dict)
    wall_adjacent: bool = False
    place_on_floor: bool = True
    offset_y: float = 0.0


@dataclass
class RoomSpec:
    width: float
    depth: float
    height: float
    room_type: str = "living_room"


WALL_NONE = 0
WALL_BACK = 1
WALL_LEFT = 2
WALL_RIGHT = 3
WALL_FRONT = 4

PRIORITY_ORDER = {
    "sofa": 1,
    "bed": 1,
    "table": 2,
    "desk": 2,
    "cabinet": 3,
    "shelf": 3,
    "chair": 4,
    "lamp": 5,
    "rug": 6,
    "decor": 7,
}

GROUPABLE_CATEGORIES = {"chair", "lamp", "decor", "table"}


def _sort_by_priority(models: list[dict]) -> list[dict]:
    def priority(m):
        cat = m.get("category", "")
        return PRIORITY_ORDER.get(cat, 99)
    return sorted(models, key=priority)


def _find_anchor(models: list[dict], category: str) -> Optional[dict]:
    for m in models:
        if m.get("category") == category:
            return m
    return None


def _group_by_original_id(models: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for m in models:
        key = m.get("original_model_id", m["id"])
        if key not in groups:
            groups[key] = []
        groups[key].append(m)
    return groups


def _check_overlap(
    pos: tuple[float, float, float],
    dims: dict,
    placed: list[Placement],
    margin: float = 0.15,
) -> bool:
    x1_min = pos[0] - dims["width"] / 2 - margin
    x1_max = pos[0] + dims["width"] / 2 + margin
    z1_min = pos[2] - dims["depth"] / 2 - margin
    z1_max = pos[2] + dims["depth"] / 2 + margin

    for p in placed:
        d = p.dimensions
        x2_min = p.position[0] - d["width"] / 2 - margin
        x2_max = p.position[0] + d["width"] / 2 + margin
        z2_min = p.position[2] - d["depth"] / 2 - margin
        z2_max = p.position[2] + d["depth"] / 2 + margin

        if not (x1_max <= x2_min or x1_min >= x2_max or
                z1_max <= z2_min or z1_min >= z2_max):
            return True
    return False


def _wall_position(
    wall: int,
    room: RoomSpec,
    dims: dict,
    placed: list[Placement],
    rng: random.Random,
) -> tuple[float, float, float]:
    hw = room.width / 2
    hd = room.depth / 2
    margin_w = dims["width"] / 2 + 0.1
    margin_d = dims["depth"] / 2 + 0.1

    if wall == WALL_BACK:
        x = rng.uniform(-hw + margin_w, hw - margin_w)
        z = -hd + margin_d
    elif wall == WALL_FRONT:
        x = rng.uniform(-hw + margin_w, hw - margin_w)
        z = hd - margin_d
    elif wall == WALL_LEFT:
        x = -hw + margin_w
        z = rng.uniform(-hd + margin_d, hd - margin_d)
    elif wall == WALL_RIGHT:
        x = hw - margin_w
        z = rng.uniform(-hd + margin_d, hd - margin_d)
    else:
        x = rng.uniform(-hw + margin_w, hw - margin_w)
        z = rng.uniform(-hd + margin_d, hd - margin_d)

    return (x, 0.0, z)


def _wall_rotation_y(wall: int) -> float:
    rotation_map = {
        WALL_BACK: 0.0,
        WALL_FRONT: math.pi,
        WALL_LEFT: math.pi / 2,
        WALL_RIGHT: -math.pi / 2,
    }
    return rotation_map.get(wall, 0.0)


def _chair_around_table(
    chair_info: dict,
    chair_idx: int,
    total_chairs: int,
    table_pos: tuple[float, float, float],
    table_dims: dict,
    room: RoomSpec,
    placed: list[Placement],
    rng: random.Random,
) -> tuple[tuple[float, float, float], float]:
    tw = table_dims.get("width", 1.0)
    td = table_dims.get("depth", 1.0)
    chair_d = chair_info.get("dimensions", {}).get("depth", 0.5)
    gap = 0.15

    if total_chairs <= 2:
        positions = [
            (table_pos[0], table_pos[1], table_pos[2] - td / 2 - chair_d / 2 - gap),
            (table_pos[0], table_pos[1], table_pos[2] + td / 2 + chair_d / 2 + gap),
        ]
        rotations = [math.pi, 0.0]
    elif total_chairs <= 4:
        positions = [
            (table_pos[0], table_pos[1], table_pos[2] - td / 2 - chair_d / 2 - gap),
            (table_pos[0], table_pos[1], table_pos[2] + td / 2 + chair_d / 2 + gap),
            (table_pos[0] - tw / 2 - chair_d / 2 - gap, table_pos[1], table_pos[2]),
            (table_pos[0] + tw / 2 + chair_d / 2 + gap, table_pos[1], table_pos[2]),
        ]
        rotations = [math.pi, 0.0, math.pi / 2, -math.pi / 2]
    else:
        positions = []
        rotations = []
        radius = max(tw, td) / 2 + chair_d + gap
        for i in range(total_chairs):
            angle = 2 * math.pi * i / total_chairs
            cx = table_pos[0] + radius * math.cos(angle)
            cz = table_pos[2] + radius * math.sin(angle)
            positions.append((cx, table_pos[1], cz))
            rotations.append(angle + math.pi)

    if chair_idx < len(positions):
        pos = positions[chair_idx]
        rot = rotations[chair_idx]
    else:
        pos = _floor_position(room, chair_info.get("dimensions", {}), placed, rng)
        rot = rng.uniform(0, 2 * math.pi)

    if _check_overlap(pos, chair_info.get("dimensions", {}), placed):
        for attempt in range(15):
            angle_offset = rng.uniform(-0.3, 0.3)
            new_angle = rotations[chair_idx % len(rotations)] + angle_offset
            r = max(tw, td) / 2 + chair_d + gap + rng.uniform(0, 0.3)
            cx = table_pos[0] + r * math.cos(new_angle - math.pi)
            cz = table_pos[2] + r * math.sin(new_angle - math.pi)
            new_pos = (cx, table_pos[1], cz)
            if not _check_overlap(new_pos, chair_info.get("dimensions", {}), placed):
                pos = new_pos
                rot = new_angle
                break

    return pos, rot


def _linear_row(
    items: list[dict],
    room: RoomSpec,
    wall: int,
    rng: random.Random,
) -> list[tuple[tuple[float, float, float], float]]:
    if not items:
        return []

    total_width = sum(i.get("dimensions", {}).get("width", 0.5) for i in items)
    total_width += 0.15 * (len(items) - 1)

    hw = room.width / 2
    hd = room.depth / 2
    first_dims = items[0].get("dimensions", {})
    margin_w = first_dims.get("width", 0.5) / 2

    if total_width > (room.width - 0.4):
        total_width = room.width - 0.4

    start_x = -total_width / 2
    results = []

    current_x = start_x
    for item in items:
        dims = item.get("dimensions", {})
        w = dims.get("width", 0.5)
        d = dims.get("depth", 0.5)
        cx = current_x + w / 2

        if wall == WALL_BACK:
            pos = (cx, 0.0, -hd + d / 2 + 0.1)
            rot = 0.0
        elif wall == WALL_FRONT:
            pos = (cx, 0.0, hd - d / 2 - 0.1)
            rot = math.pi
        elif wall == WALL_LEFT:
            pos = (-hw + d / 2 + 0.1, 0.0, cx)
            rot = math.pi / 2
        elif wall == WALL_RIGHT:
            pos = (hw - d / 2 - 0.1, 0.0, cx)
            rot = -math.pi / 2
        else:
            pos = (cx, 0.0, rng.uniform(-hd + d / 2, hd - d / 2))
            rot = rng.uniform(0, 2 * math.pi)

        results.append((pos, rot))
        current_x += w + 0.15

    return results


def compute_layout(
    models: list[dict],
    room_type: str = "living_room",
    room_width: Optional[float] = None,
    room_depth: Optional[float] = None,
    room_height: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict:
    room = RoomSpec(
        width=room_width or settings.default_room_width,
        depth=room_depth or settings.default_room_depth,
        height=room_height or settings.default_room_height,
        room_type=room_type,
    )

    rng = random.Random(seed)
    sorted_models = _sort_by_priority(models)
    placements: list[Placement] = []
    model_placements: dict[str, Placement] = {}

    wall_assignments = _assign_walls(sorted_models, room, rng)

    tables = [m for m in sorted_models if m.get("category") == "table"]
    chairs = [m for m in sorted_models if m.get("category") == "chair"]
    others = [m for m in sorted_models if m.get("category") not in {"table", "chair"}]

    for table_info in tables:
        dims = table_info.get("dimensions", {"width": 1, "height": 1, "depth": 1})
        wall_adj = table_info.get("wall_adjacent", False)
        place_floor = table_info.get("place_on_floor", True)
        offset_y = table_info.get("offset_y", 0.0)
        wall = wall_assignments.get(table_info["id"], WALL_NONE)

        if wall != WALL_NONE and wall_adj:
            pos = _wall_position(wall, room, dims, placements, rng)
            rot_y = _wall_rotation_y(wall)
        else:
            pos = _floor_position(room, dims, placements, rng)
            rot_y = rng.uniform(0, 2 * math.pi)

        for attempt in range(10):
            if not _check_overlap(pos, dims, placements):
                break
            pos = _floor_position(room, dims, placements, rng)
            rot_y = rng.uniform(0, 2 * math.pi)

        if place_floor:
            y = offset_y + dims.get("height", 1.0) / 2
        else:
            y = offset_y

        placement = Placement(
            model_id=table_info["id"],
            position=(pos[0], y, pos[2]),
            rotation_y=rot_y,
            dimensions=dims,
            wall_adjacent=wall_adj,
            place_on_floor=place_floor,
            offset_y=offset_y,
        )
        placements.append(placement)
        model_placements[table_info["id"]] = placement
        logger.info(
            "Placed table %s at (%.2f, %.2f, %.2f) rot=%.1f°",
            table_info["id"], pos[0], y, pos[2], math.degrees(rot_y),
        )

    chair_groups = _group_by_original_id(chairs)
    processed_chairs = set()

    for table_info in tables:
        table_placement = model_placements.get(table_info["id"])
        if not table_placement or not chairs:
            continue

        table_pos = table_placement.position
        table_dims = table_info.get("dimensions", {"width": 1, "depth": 1})

        for orig_id, chair_group in chair_groups.items():
            if orig_id in processed_chairs:
                continue
            if len(chair_group) <= 0:
                continue

            chairs_for_table = min(len(chair_group), 6)
            for i in range(chairs_for_table):
                chair_info = chair_group[i]
                if chair_info["id"] in processed_chairs:
                    continue

                dims = chair_info.get("dimensions", {"width": 0.5, "height": 1, "depth": 0.5})
                place_floor = chair_info.get("place_on_floor", True)
                offset_y = chair_info.get("offset_y", 0.0)

                pos, rot_y = _chair_around_table(
                    chair_info, i, chairs_for_table,
                    table_pos, table_dims,
                    room, placements, rng,
                )

                if _check_overlap(pos, dims, placements):
                    pos = _floor_position(room, dims, placements, rng)
                    rot_y = rng.uniform(0, 2 * math.pi)
                    for attempt in range(15):
                        if not _check_overlap(pos, dims, placements):
                            break
                        pos = _floor_position(room, dims, placements, rng)
                        rot_y = rng.uniform(0, 2 * math.pi)

                if place_floor:
                    y = offset_y + dims.get("height", 1.0) / 2
                else:
                    y = offset_y

                placement = Placement(
                    model_id=chair_info["id"],
                    position=(pos[0], y, pos[2]),
                    rotation_y=rot_y,
                    dimensions=dims,
                    wall_adjacent=False,
                    place_on_floor=place_floor,
                    offset_y=offset_y,
                )
                placements.append(placement)
                model_placements[chair_info["id"]] = placement
                processed_chairs.add(chair_info["id"])
                logger.info(
                    "Placed chair %s around table at (%.2f, %.2f, %.2f) rot=%.1f°",
                    chair_info["id"], pos[0], y, pos[2], math.degrees(rot_y),
                )
            processed_chairs.add(orig_id)

    for c in chairs:
        if c["id"] not in processed_chairs:
            others.append(c)

    sorted_others = _sort_by_priority(others)
    other_groups = _group_by_original_id(sorted_others)

    for orig_id, group in other_groups.items():
        if len(group) <= 1:
            for model_info in group:
                _place_single_model(
                    model_info, room, rng,
                    placements, model_placements, wall_assignments,
                )
        else:
            first = group[0]
            wall = wall_assignments.get(first["id"], WALL_NONE)
            wall_adj = first.get("wall_adjacent", False)

            if wall_adj and wall != WALL_NONE:
                positions = _linear_row(group, room, wall, rng)
                for i, model_info in enumerate(group):
                    dims = model_info.get("dimensions", {"width": 1, "height": 1, "depth": 1})
                    place_floor = model_info.get("place_on_floor", True)
                    offset_y = model_info.get("offset_y", 0.0)

                    if i < len(positions):
                        pos, rot_y = positions[i]
                    else:
                        pos = _floor_position(room, dims, placements, rng)
                        rot_y = rng.uniform(0, 2 * math.pi)

                    for attempt in range(10):
                        if not _check_overlap(pos, dims, placements):
                            break
                        pos = _floor_position(room, dims, placements, rng)
                        rot_y = rng.uniform(0, 2 * math.pi)

                    if place_floor:
                        y = offset_y + dims.get("height", 1.0) / 2
                    else:
                        y = offset_y

                    placement = Placement(
                        model_id=model_info["id"],
                        position=(pos[0], y, pos[2]),
                        rotation_y=rot_y,
                        dimensions=dims,
                        wall_adjacent=wall_adj,
                        place_on_floor=place_floor,
                        offset_y=offset_y,
                    )
                    placements.append(placement)
                    model_placements[model_info["id"]] = placement
                    logger.info(
                        "Placed grouped %s at (%.2f, %.2f, %.2f)",
                        model_info["id"], pos[0], y, pos[2],
                    )
            else:
                for i, model_info in enumerate(group):
                    _place_single_model(
                        model_info, room, rng,
                        placements, model_placements, wall_assignments,
                    )

    return {
        "room": {
            "width": room.width,
            "depth": room.depth,
            "height": room.height,
            "type": room.room_type,
        },
        "placements": [
            {
                "model_id": p.model_id,
                "position": list(p.position),
                "rotation_y": p.rotation_y,
                "scale": list(p.scale),
                "dimensions": p.dimensions,
                "wall_adjacent": p.wall_adjacent,
            }
            for p in placements
        ],
    }


def _place_single_model(
    model_info: dict,
    room: RoomSpec,
    rng: random.Random,
    placements: list[Placement],
    model_placements: dict[str, Placement],
    wall_assignments: dict[str, int],
):
    dims = model_info.get("dimensions", {"width": 1, "height": 1, "depth": 1})
    wall_adj = model_info.get("wall_adjacent", False)
    place_floor = model_info.get("place_on_floor", True)
    offset_y = model_info.get("offset_y", 0.0)

    wall = wall_assignments.get(model_info["id"], WALL_NONE)

    if wall != WALL_NONE and wall_adj:
        pos = _wall_position(wall, room, dims, placements, rng)
        rot_y = _wall_rotation_y(wall)
    else:
        pos = _floor_position(room, dims, placements, rng)
        rot_y = rng.uniform(0, 2 * math.pi)

    for attempt in range(15):
        if not _check_overlap(pos, dims, placements):
            break
        if wall != WALL_NONE and wall_adj:
            pos = _wall_position(wall, room, dims, placements, rng)
        else:
            pos = _floor_position(room, dims, placements, rng)
            rot_y = rng.uniform(0, 2 * math.pi)

    if place_floor:
        y = offset_y + dims.get("height", 1.0) / 2
    else:
        y = offset_y

    placement = Placement(
        model_id=model_info["id"],
        position=(pos[0], y, pos[2]),
        rotation_y=rot_y,
        dimensions=dims,
        wall_adjacent=wall_adj,
        place_on_floor=place_floor,
        offset_y=offset_y,
    )
    placements.append(placement)
    model_placements[model_info["id"]] = placement
    logger.info(
        "Placed %s at (%.2f, %.2f, %.2f) rot=%.1f° wall=%s",
        model_info["id"], pos[0], y, pos[2],
        math.degrees(rot_y), wall,
    )


def _assign_walls(
    models: list[dict],
    room: RoomSpec,
    rng: random.Random,
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    wall_usage: dict[int, float] = {
        WALL_BACK: 0.0,
        WALL_LEFT: 0.0,
        WALL_RIGHT: 0.0,
        WALL_FRONT: 0.0,
    }

    seen_originals: set[str] = set()
    for m in models:
        if not m.get("wall_adjacent", False):
            assignments[m["id"]] = WALL_NONE
            continue

        orig_id = m.get("original_model_id", m["id"])
        if orig_id in seen_originals:
            chosen = assignments.get(
                next(k for k, v in assignments.items()
                     if orig_id in k),
                WALL_NONE,
            )
            assignments[m["id"]] = chosen
            continue

        available_walls = [
            w for w, usage in wall_usage.items()
            if usage < room.width * 0.7
        ]
        if not available_walls:
            available_walls = list(wall_usage.keys())

        chosen = min(available_walls, key=lambda w: wall_usage[w])
        assignments[m["id"]] = chosen
        seen_originals.add(orig_id)

        total_width = 0
        orig_group = [
            x for x in models
            if x.get("original_model_id", x["id"]) == orig_id
        ]
        for gm in orig_group:
            dims = gm.get("dimensions", {"width": 1.0})
            total_width += dims.get("width", 1.0) + 0.15
        wall_usage[chosen] += total_width

    return assignments


def _floor_position(
    room: RoomSpec,
    dims: dict,
    placed: list[Placement],
    rng: random.Random,
) -> tuple[float, float, float]:
    hw = room.width / 2
    hd = room.depth / 2
    margin_w = dims.get("width", 1.0) / 2 + 0.2
    margin_d = dims.get("depth", 1.0) / 2 + 0.2

    x = rng.uniform(-hw + margin_w, hw - margin_w)
    z = rng.uniform(-hd + margin_d, hd - margin_d)

    return (x, 0.0, z)
