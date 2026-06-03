import logging
from pathlib import Path
from typing import Optional

from app.services.model_library import ModelLibrary, get_model_library
from app.services.model_retrieval import retrieve_models, detect_room_type
from app.services.layout_inference import compute_layout
from app.services.scene_exporter import export_scene
from app.services.rendering_service import render_scene

logger = logging.getLogger(__name__)


class Text2RoomPipeline:
    def __init__(self, library: Optional[ModelLibrary] = None):
        self.library = library or get_model_library()

    def run(
        self,
        text: str,
        room_width: Optional[float] = None,
        room_depth: Optional[float] = None,
        room_height: Optional[float] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        seed: Optional[int] = None,
        render: bool = True,
        render_width: int = 1280,
        render_height: int = 720,
        render_views: int = 4,
    ) -> dict:
        logger.info("Pipeline started for: '%s'", text)

        room_type = detect_room_type(text)
        logger.info("Detected room type: %s", room_type)

        matched_models = retrieve_models(
            text=text,
            library=self.library,
            top_k=top_k,
            threshold=similarity_threshold,
        )
        logger.info("Matched %d models (including quantity instances)", len(matched_models))

        unique_models: dict[str, dict] = {}
        for m in matched_models:
            orig_id = m.get("original_model_id", m["id"])
            if orig_id not in unique_models:
                unique_models[orig_id] = {
                    "id": orig_id,
                    "category": m.get("category"),
                    "description": m.get("description"),
                    "similarity": m.get("similarity"),
                    "quantity": m.get("requested_quantity", 1),
                }

        if not matched_models:
            logger.warning("No models matched, returning empty scene")
            return {
                "status": "no_matches",
                "text": text,
                "room_type": room_type,
                "matched_models": [],
                "layout": None,
                "output_path": None,
                "render_paths": [],
            }

        layout = compute_layout(
            models=matched_models,
            room_type=room_type,
            room_width=room_width,
            room_depth=room_depth,
            room_height=room_height,
            seed=seed,
        )
        logger.info("Layout computed with %d placements", len(layout["placements"]))

        model_infos: dict[str, dict] = {}
        for m in matched_models:
            orig_id = m.get("original_model_id", m["id"])
            inst_id = m["id"]
            model_infos[inst_id] = {
                "id": orig_id,
                "instance_id": inst_id,
                "category": m.get("category"),
                "description": m.get("description"),
                "similarity": m.get("similarity"),
                "original_model_id": orig_id,
                "quantity_index": m.get("quantity_index"),
                "quantity_total": m.get("quantity_total"),
            }
        output_path = export_scene(layout=layout, model_infos=model_infos)
        logger.info("Scene exported to: %s", output_path)

        render_paths = []
        if render:
            try:
                scene_dir = str(Path(output_path).parent / "renders")
                render_paths = render_scene(
                    layout=layout,
                    model_infos=model_infos,
                    output_dir=scene_dir,
                    width=render_width,
                    height=render_height,
                    n_views=render_views,
                )
                logger.info("Rendered %d preview images", len(render_paths))
            except Exception as e:
                logger.error("Rendering failed: %s", e, exc_info=True)

        return {
            "status": "success",
            "text": text,
            "room_type": room_type,
            "matched_models": list(unique_models.values()),
            "layout": layout,
            "output_path": output_path,
            "render_paths": render_paths,
        }


_pipeline: Optional[Text2RoomPipeline] = None


def get_pipeline() -> Text2RoomPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Text2RoomPipeline()
    return _pipeline
