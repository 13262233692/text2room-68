import json
import logging
import math
import struct
import uuid
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


def _make_gltf_id() -> str:
    return str(uuid.uuid4())[:8]


def float_to_bytes(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def uint_to_bytes(*values: int) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)


def _create_room_mesh(room: dict) -> dict:
    w = room["width"] / 2
    d = room["depth"] / 2
    h = room["height"]

    floor_verts = [
        -w, 0, -d,
        w, 0, -d,
        w, 0, d,
        -w, 0, d,
    ]
    floor_indices = [0, 2, 1, 0, 3, 2]

    ceiling_verts = [
        -w, h, -d,
        w, h, -d,
        w, h, d,
        -w, h, d,
    ]
    ceiling_indices = [0, 1, 2, 0, 2, 3]

    wall_back_verts = [
        -w, 0, -d,
        w, 0, -d,
        w, h, -d,
        -w, h, -d,
    ]
    wall_back_indices = [0, 1, 2, 0, 2, 3]

    wall_front_verts = [
        -w, 0, d,
        w, 0, d,
        w, h, d,
        -w, h, d,
    ]
    wall_front_indices = [0, 2, 1, 0, 3, 2]

    wall_left_verts = [
        -w, 0, -d,
        -w, 0, d,
        -w, h, d,
        -w, h, -d,
    ]
    wall_left_indices = [0, 1, 2, 0, 2, 3]

    wall_right_verts = [
        w, 0, -d,
        w, 0, d,
        w, h, d,
        w, h, -d,
    ]
    wall_right_indices = [0, 2, 1, 0, 3, 2]

    all_verts = (
        floor_verts + ceiling_verts +
        wall_back_verts + wall_front_verts +
        wall_left_verts + wall_right_verts
    )

    normals = [0, 1, 0] * 4 + [0, -1, 0] * 4
    normals += [0, 0, 1] * 4 + [0, 0, -1] * 4
    normals += [1, 0, 0] * 4 + [-1, 0, 0] * 4

    vert_count = 4
    idx_offset = 0
    all_indices = []
    for indices in [floor_indices, ceiling_indices, wall_back_indices,
                    wall_front_indices, wall_left_indices, wall_right_indices]:
        all_indices.extend([i + idx_offset for i in indices])
        idx_offset += vert_count

    positions_bin = float_to_bytes(*all_verts)
    normals_bin = float_to_bytes(*normals)
    indices_bin = uint_to_bytes(*all_indices)

    total_verts = 24
    pos_byte_len = len(positions_bin)
    norm_byte_len = len(normals_bin)
    idx_byte_len = len(indices_bin)

    buffer_content = positions_bin + normals_bin + indices_bin
    buffer_uri = "room.bin"

    gltf = {
        "asset": {"version": "2.0", "generator": "text2room"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"mesh": 0, "name": "Room"},
        ],
        "meshes": [
            {
                "name": "RoomMesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ],
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": total_verts,
                "type": "VEC3",
                "max": [w, h, d],
                "min": [-w, 0, -d],
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": total_verts,
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": len(all_indices),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_byte_len, "target": 34962},
            {"buffer": 0, "byteOffset": pos_byte_len, "byteLength": norm_byte_len, "target": 34962},
            {"buffer": 0, "byteOffset": pos_byte_len + norm_byte_len, "byteLength": idx_byte_len, "target": 34963},
        ],
        "buffers": [
            {"uri": buffer_uri, "byteLength": len(buffer_content)},
        ],
        "materials": [
            {
                "name": "RoomMaterial",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.9, 0.88, 0.85, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
            }
        ],
    }

    return gltf, buffer_content, buffer_uri


def _create_furniture_node(placement: dict, node_index: int) -> dict:
    pos = placement["position"]
    rot_y = placement["rotation_y"]
    scale = placement.get("scale", [1.0, 1.0, 1.0])

    cos_r = math.cos(rot_y / 2)
    sin_r = math.sin(rot_y / 2)

    node = {
        "name": placement["model_id"],
        "translation": [pos[0], pos[1], pos[2]],
        "rotation": [0.0, sin_r, 0.0, cos_r],
        "scale": scale,
        "children": [],
    }

    dims = placement.get("dimensions", {"width": 1, "height": 1, "depth": 1})
    mesh_node = _create_box_mesh_node(dims, node_index)
    return node, mesh_node


def _create_box_mesh_node(dims: dict, base_idx: int) -> dict:
    hw = dims.get("width", 1.0) / 2
    hd = dims.get("depth", 1.0) / 2
    hh = dims.get("height", 1.0) / 2

    vertices = [
        -hw, -hh, -hd,  hw, -hh, -hd,  hw, hh, -hd,  -hw, hh, -hd,
        -hw, -hh, hd,   hw, -hh, hd,   hw, hh, hd,   -hw, hh, hd,
    ]

    indices = [
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        2, 3, 7, 2, 7, 6,
        0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ]

    normals = [
        0, 0, -1,  0, 0, -1,  0, 0, -1,  0, 0, -1,
        0, 0, 1,   0, 0, 1,   0, 0, 1,   0, 0, 1,
        0, -1, 0,  0, -1, 0,  0, -1, 0,  0, -1, 0,
        0, 1, 0,   0, 1, 0,   0, 1, 0,   0, 1, 0,
        -1, 0, 0,  -1, 0, 0,  -1, 0, 0,  -1, 0, 0,
        1, 0, 0,   1, 0, 0,   1, 0, 0,   1, 0, 0,
    ]

    return {
        "vertices": vertices,
        "indices": indices,
        "normals": normals,
        "dims": dims,
    }


_CATEGORY_COLORS = {
    "sofa": [0.2, 0.4, 0.8, 1.0],
    "chair": [0.6, 0.3, 0.1, 1.0],
    "table": [0.6, 0.4, 0.2, 1.0],
    "bed": [0.8, 0.7, 0.6, 1.0],
    "cabinet": [0.5, 0.35, 0.2, 1.0],
    "shelf": [0.5, 0.35, 0.2, 1.0],
    "lamp": [0.9, 0.9, 0.7, 1.0],
    "rug": [0.7, 0.5, 0.3, 1.0],
    "decor": [0.3, 0.7, 0.3, 1.0],
    "default": [0.5, 0.5, 0.5, 1.0],
}


def export_scene(
    layout: dict,
    model_infos: dict[str, dict],
    output_path: Optional[str] = None,
) -> str:
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_id = _make_gltf_id()
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    room = layout["room"]
    placements = layout["placements"]

    room_gltf, room_buffer, room_buffer_uri = _create_room_mesh(room)

    with open(scene_dir / room_buffer_uri, "wb") as f:
        f.write(room_buffer)

    all_buffers = room_gltf["buffers"]
    all_buffer_views = room_gltf["bufferViews"]
    all_accessors = room_gltf["accessors"]
    all_meshes = room_gltf["meshes"]
    all_materials = room_gltf["materials"]
    all_nodes = room_gltf["nodes"]
    scene_children = [0]

    current_buffer_idx = 1
    current_accessor_idx = len(room_gltf["accessors"])
    current_buffer_view_idx = len(room_gltf["bufferViews"])
    current_mesh_idx = 1
    current_material_idx = 1
    current_node_idx = 1

    for placement in placements:
        model_id = placement["model_id"]
        info = model_infos.get(model_id, {})
        category = info.get("category", "default")
        color = _CATEGORY_COLORS.get(category, _CATEGORY_COLORS["default"])

        box_data = _create_box_mesh_node(
            placement.get("dimensions", {"width": 1, "height": 1, "depth": 1}),
            current_node_idx,
        )

        vertices = box_data["vertices"]
        indices = box_data["indices"]
        normals = box_data["normals"]
        vert_count = len(vertices) // 3
        idx_count = len(indices)

        pos_bytes = float_to_bytes(*vertices)
        norm_bytes = float_to_bytes(*normals)
        idx_bytes = uint_to_bytes(*indices)

        buffer_content = pos_bytes + norm_bytes + idx_bytes
        buffer_uri = f"{model_id}.bin"

        with open(scene_dir / buffer_uri, "wb") as f:
            f.write(buffer_content)

        all_buffers.append({
            "uri": buffer_uri,
            "byteLength": len(buffer_content),
        })

        pos_bv_idx = current_buffer_view_idx
        norm_bv_idx = current_buffer_view_idx + 1
        idx_bv_idx = current_buffer_view_idx + 2

        all_buffer_views.extend([
            {"buffer": current_buffer_idx, "byteOffset": 0,
             "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": current_buffer_idx, "byteOffset": len(pos_bytes),
             "byteLength": len(norm_bytes), "target": 34962},
            {"buffer": current_buffer_idx,
             "byteOffset": len(pos_bytes) + len(norm_bytes),
             "byteLength": len(idx_bytes), "target": 34963},
        ])

        dims = box_data["dims"]
        all_accessors.extend([
            {
                "bufferView": pos_bv_idx,
                "componentType": 5126,
                "count": vert_count,
                "type": "VEC3",
                "max": [dims["width"] / 2, dims["height"] / 2, dims["depth"] / 2],
                "min": [-dims["width"] / 2, -dims["height"] / 2, -dims["depth"] / 2],
            },
            {
                "bufferView": norm_bv_idx,
                "componentType": 5126,
                "count": vert_count,
                "type": "VEC3",
            },
            {
                "bufferView": idx_bv_idx,
                "componentType": 5125,
                "count": idx_count,
                "type": "SCALAR",
            },
        ])

        all_materials.append({
            "name": f"{model_id}_material",
            "pbrMetallicRoughness": {
                "baseColorFactor": color,
                "metallicFactor": 0.1,
                "roughnessFactor": 0.7,
            },
        })

        all_meshes.append({
            "name": f"{model_id}_mesh",
            "primitives": [
                {
                    "attributes": {
                        "POSITION": current_accessor_idx,
                        "NORMAL": current_accessor_idx + 1,
                    },
                    "indices": current_accessor_idx + 2,
                    "material": current_material_idx,
                }
            ],
        })

        pos = placement["position"]
        rot_y = placement["rotation_y"]
        scale = placement.get("scale", [1.0, 1.0, 1.0])
        cos_r = math.cos(rot_y / 2)
        sin_r = math.sin(rot_y / 2)

        mesh_node_idx = current_node_idx
        all_nodes.append({
            "name": f"{model_id}_mesh",
            "mesh": current_mesh_idx,
        })

        transform_node_idx = current_node_idx + 1
        all_nodes.append({
            "name": model_id,
            "translation": [pos[0], pos[1], pos[2]],
            "rotation": [0.0, sin_r, 0.0, cos_r],
            "scale": scale,
            "children": [mesh_node_idx],
        })

        scene_children.append(transform_node_idx)

        current_buffer_idx += 1
        current_buffer_view_idx += 3
        current_accessor_idx += 3
        current_mesh_idx += 1
        current_material_idx += 1
        current_node_idx += 2

    gltf = {
        "asset": {"version": "2.0", "generator": "text2room"},
        "scene": 0,
        "scenes": [{"nodes": scene_children}],
        "nodes": all_nodes,
        "meshes": all_meshes,
        "accessors": all_accessors,
        "bufferViews": all_buffer_views,
        "buffers": all_buffers,
        "materials": all_materials,
    }

    gltf_path = scene_dir / "scene.gltf"
    with open(gltf_path, "w", encoding="utf-8") as f:
        json.dump(gltf, f, indent=2, ensure_ascii=False)

    metadata = {
        "scene_id": scene_id,
        "room": room,
        "placements": placements,
        "model_infos": {k: {
            "id": v.get("id"),
            "category": v.get("category"),
            "description": v.get("description"),
            "similarity": v.get("similarity"),
        } for k, v in model_infos.items()},
    }
    with open(scene_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Scene exported to %s", scene_dir)
    return str(gltf_path)
