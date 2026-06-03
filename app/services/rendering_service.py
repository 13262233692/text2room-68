import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import pyrender
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

_CATEGORY_PBR = {
    "sofa": {
        "base_color": [0.2, 0.4, 0.8, 1.0],
        "metallic": 0.0,
        "roughness": 0.85,
    },
    "chair": {
        "base_color": [0.55, 0.3, 0.1, 1.0],
        "metallic": 0.05,
        "roughness": 0.6,
    },
    "table": {
        "base_color": [0.6, 0.4, 0.2, 1.0],
        "metallic": 0.02,
        "roughness": 0.5,
    },
    "bed": {
        "base_color": [0.85, 0.75, 0.65, 1.0],
        "metallic": 0.0,
        "roughness": 0.9,
    },
    "cabinet": {
        "base_color": [0.5, 0.35, 0.2, 1.0],
        "metallic": 0.1,
        "roughness": 0.45,
    },
    "shelf": {
        "base_color": [0.5, 0.35, 0.2, 1.0],
        "metallic": 0.1,
        "roughness": 0.45,
    },
    "lamp": {
        "base_color": [0.95, 0.95, 0.8, 1.0],
        "metallic": 0.3,
        "roughness": 0.3,
        "emissive": [1.0, 0.95, 0.8],
        "emissive_intensity": 2.0,
    },
    "rug": {
        "base_color": [0.7, 0.5, 0.3, 1.0],
        "metallic": 0.0,
        "roughness": 1.0,
    },
    "decor": {
        "base_color": [0.3, 0.7, 0.3, 1.0],
        "metallic": 0.0,
        "roughness": 0.7,
    },
    "default": {
        "base_color": [0.5, 0.5, 0.5, 1.0],
        "metallic": 0.0,
        "roughness": 0.7,
    },
}

_ROOM_WALL_COLOR = [0.92, 0.90, 0.87, 1.0]
_ROOM_FLOOR_COLOR = [0.65, 0.58, 0.50, 1.0]


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)

    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, fwd)
    true_up = true_up / np.linalg.norm(true_up)

    pose = np.eye(4)
    pose[0, :3] = right
    pose[1, :3] = true_up
    pose[2, :3] = -fwd
    pose[:3, 3] = eye
    return pose


def _create_room_meshes(room: dict) -> list:
    hw = room["width"] / 2
    hd = room["depth"] / 2
    h = room["height"]
    meshes = []

    floor = trimesh.creation.box(extents=[room["width"], 0.02, room["depth"]])
    floor.apply_translation([0, -0.01, 0])
    floor.visual = trimesh.visual.ColorVisuals(
        floor,
        vertex_colors=[int(c * 255) for c in _ROOM_FLOOR_COLOR[:3]] + [255],
    )
    meshes.append(("floor", floor))

    wall_back = trimesh.creation.box(extents=[room["width"], h, 0.05])
    wall_back.apply_translation([0, h / 2, -hd])
    wall_back.visual = trimesh.visual.ColorVisuals(
        wall_back,
        vertex_colors=[int(c * 255) for c in _ROOM_WALL_COLOR[:3]] + [255],
    )
    meshes.append(("wall_back", wall_back))

    wall_front = trimesh.creation.box(extents=[room["width"], h, 0.05])
    wall_front.apply_translation([0, h / 2, hd])
    wall_front.visual = trimesh.visual.ColorVisuals(
        wall_front,
        vertex_colors=[int(c * 255) for c in _ROOM_WALL_COLOR[:3]] + [255],
    )
    meshes.append(("wall_front", wall_front))

    wall_left = trimesh.creation.box(extents=[0.05, h, room["depth"]])
    wall_left.apply_translation([-hw, h / 2, 0])
    wall_left.visual = trimesh.visual.ColorVisuals(
        wall_left,
        vertex_colors=[int(c * 255) for c in _ROOM_WALL_COLOR[:3]] + [255],
    )
    meshes.append(("wall_left", wall_left))

    wall_right = trimesh.creation.box(extents=[0.05, h, room["depth"]])
    wall_right.apply_translation([hw, h / 2, 0])
    wall_right.visual = trimesh.visual.ColorVisuals(
        wall_right,
        vertex_colors=[int(c * 255) for c in _ROOM_WALL_COLOR[:3]] + [255],
    )
    meshes.append(("wall_right", wall_right))

    return meshes


def _create_furniture_mesh(placement: dict, category: str) -> trimesh.Trimesh:
    dims = placement.get("dimensions", {"width": 1, "height": 1, "depth": 1})
    w = dims.get("width", 1.0)
    h = dims.get("height", 1.0)
    d = dims.get("depth", 1.0)

    mesh = trimesh.creation.box(extents=[w, h, d])

    pbr = _CATEGORY_PBR.get(category, _CATEGORY_PBR["default"])
    base = pbr["base_color"]
    vertex_color = [int(c * 255) for c in base[:3]] + [255]
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, vertex_colors=vertex_color
    )

    pos = placement["position"]
    rot_y = placement["rotation_y"]

    transform = np.eye(4)
    transform[0, 0] = math.cos(rot_y)
    transform[0, 2] = math.sin(rot_y)
    transform[2, 0] = -math.sin(rot_y)
    transform[2, 2] = math.cos(rot_y)
    transform[:3, 3] = [pos[0], pos[1], pos[2]]
    mesh.apply_transform(transform)

    return mesh


def _add_lights(scene: pyrender.Scene, room: dict) -> None:
    h = room["height"]

    ceiling_light = pyrender.PointLight(
        color=[1.0, 0.98, 0.95],
        intensity=8.0,
    )
    ceiling_pose = np.eye(4)
    ceiling_pose[1, 3] = h - 0.1
    scene.add(ceiling_light, pose=ceiling_pose, name="ceiling_light")

    fill1 = pyrender.DirectionalLight(
        color=[0.85, 0.88, 1.0],
        intensity=2.5,
    )
    fill1_pose = np.eye(4)
    fill1_pose[:3, 3] = [2.0, h - 0.3, -1.0]
    fwd = np.array([-0.3, -0.7, 0.2])
    fwd = fwd / np.linalg.norm(fwd)
    fill1_pose[2, :3] = -fwd
    scene.add(fill1, pose=fill1_pose, name="fill_light_1")

    fill2 = pyrender.DirectionalLight(
        color=[1.0, 0.95, 0.9],
        intensity=1.5,
    )
    fill2_pose = np.eye(4)
    fill2_pose[:3, 3] = [-2.0, h - 0.3, 1.0]
    fwd = np.array([0.3, -0.7, -0.2])
    fwd = fwd / np.linalg.norm(fwd)
    fill2_pose[2, :3] = -fwd
    scene.add(fill2, pose=fill2_pose, name="fill_light_2")

    ambient = pyrender.DirectionalLight(
        color=[0.9, 0.9, 0.95],
        intensity=0.8,
    )
    ambient_pose = np.eye(4)
    ambient_pose[1, 3] = h / 2
    scene.add(ambient, pose=ambient_pose, name="ambient_light")


def _add_lamp_lights(
    scene: pyrender.Scene,
    placements: list[dict],
    model_infos: dict,
) -> None:
    for p in placements:
        model_id = p["model_id"]
        info = model_infos.get(model_id, {})
        category = info.get("category", "")
        if category != "lamp":
            continue

        pbr = _CATEGORY_PBR.get("lamp", _CATEGORY_PBR["default"])
        emissive = pbr.get("emissive", [1.0, 0.95, 0.8])
        intensity = pbr.get("emissive_intensity", 2.0)

        pos = p["position"]
        dims = p.get("dimensions", {"height": 1.0})
        lamp_h = dims.get("height", 1.0)

        light = pyrender.PointLight(
            color=emissive + [1.0],
            intensity=intensity,
        )
        light_pose = np.eye(4)
        light_pose[:3, 3] = [pos[0], pos[1] + lamp_h * 0.4, pos[2]]
        scene.add(light, pose=light_pose, name=f"lamp_light_{model_id}")


def _compute_camera_poses(room: dict, n_views: int = 4) -> list[np.ndarray]:
    hw = room["width"] / 2
    hd = room["depth"] / 2
    h = room["height"]
    center = np.array([0.0, h * 0.4, 0.0])
    wall_margin = 0.35

    eye_positions = [
        np.array([0.0, h * 0.55, hd - wall_margin]),
        np.array([0.0, h * 0.55, -hd + wall_margin]),
        np.array([hw - wall_margin, h * 0.55, 0.0]),
        np.array([-hw + wall_margin, h * 0.55, 0.0]),
    ]

    poses = []
    for i in range(min(n_views, len(eye_positions))):
        pose = _look_at(eye_positions[i], center)
        poses.append(pose)

    if n_views > len(eye_positions):
        for angle_deg in [30, 60, -30, -60]:
            angle = math.radians(angle_deg)
            r = max(hw, hd) * 0.6
            eye = np.array([
                r * math.sin(angle),
                h * 0.55,
                r * math.cos(angle),
            ])
            pose = _look_at(eye, center)
            poses.append(pose)
            if len(poses) >= n_views:
                break

    return poses[:n_views]


def render_scene(
    layout: dict,
    model_infos: dict,
    output_dir: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    n_views: int = 4,
) -> list[str]:
    import os
    if "PYOPENGL_PLATFORM" in os.environ:
        del os.environ["PYOPENGL_PLATFORM"]

    room = layout["room"]
    placements = layout["placements"]

    scene = pyrender.Scene(
        bg_color=[0.15, 0.15, 0.18, 1.0],
        ambient_light=[0.15, 0.15, 0.17],
    )

    room_meshes = _create_room_meshes(room)
    for name, mesh in room_meshes:
        prim = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(prim, name=name)

    for placement in placements:
        model_id = placement["model_id"]
        info = model_infos.get(model_id, {})
        category = info.get("category", "default")

        furniture_mesh = _create_furniture_mesh(placement, category)
        prim = pyrender.Mesh.from_trimesh(furniture_mesh, smooth=True)

        pbr = _CATEGORY_PBR.get(category, _CATEGORY_PBR["default"])
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=pbr["base_color"],
            metallicFactor=pbr["metallic"],
            roughnessFactor=pbr["roughness"],
            alphaMode="OPAQUE",
            emissiveFactor=pbr.get("emissive", [0.0, 0.0, 0.0]),
        )
        prim.primitives[0].material = material

        scene.add(prim, name=f"furniture_{model_id}")

    _add_lights(scene, room)
    _add_lamp_lights(scene, placements, model_infos)

    camera = pyrender.PerspectiveCamera(yfov=math.pi / 3.0, znear=0.05, zfar=50.0)
    cam_node = scene.add(camera, name="main_camera")

    if output_dir is None:
        output_dir = str(settings.output_dir / "renders")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    camera_poses = _compute_camera_poses(room, n_views)

    renderer = pyrender.OffscreenRenderer(width, height)

    rendered_paths = []
    for i, cam_pose in enumerate(camera_poses):
        scene.set_pose(cam_node, pose=cam_pose)

        color, depth = renderer.render(scene)

        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            near = max(0.1, float(np.percentile(valid_depth, 5)))
            far = min(float(np.percentile(valid_depth, 95)) * 1.5, 30.0)
            if far > near + 0.5:
                shadow_factor = np.clip((depth - near) / (far - near), 0, 1)
                shadow_factor = 1.0 - shadow_factor * 0.3
                color = np.clip(
                    color.astype(np.float32) * shadow_factor[:, :, np.newaxis],
                    0, 255,
                ).astype(np.uint8)

        img = Image.fromarray(color)
        filename = f"view_{i:02d}.png"
        filepath = out_path / filename
        img.save(str(filepath))

        rendered_paths.append(str(filepath))
        logger.info("Rendered view %d to %s", i, filepath)

    renderer.delete()

    logger.info("Rendered %d views to %s", len(rendered_paths), out_path)
    return rendered_paths


def render_single_view(
    layout: dict,
    model_infos: dict,
    output_path: str,
    width: int = 1280,
    height: int = 720,
    view_index: int = 0,
) -> str:
    paths = render_scene(
        layout=layout,
        model_infos=model_infos,
        output_dir=str(Path(output_path).parent),
        width=width,
        height=height,
        n_views=view_index + 1,
    )
    if view_index < len(paths):
        src = Path(paths[view_index])
        dst = Path(output_path)
        if src != dst:
            import shutil
            shutil.move(str(src), str(dst))
        return str(dst)
    return paths[-1] if paths else ""
