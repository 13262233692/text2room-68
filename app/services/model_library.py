import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.config import settings
from app.services.text_encoder import encode_text

logger = logging.getLogger(__name__)

_METADATA_FILE = "metadata.json"
_EMBEDDINGS_FILE = "embeddings.pt"


class ModelLibrary:
    def __init__(self, library_dir: Optional[Path] = None):
        self.library_dir = library_dir or settings.model_library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: dict[str, dict] = {}
        self.embeddings: dict[str, torch.Tensor] = {}
        self._load()

    def _metadata_path(self) -> Path:
        return self.library_dir / _METADATA_FILE

    def _embeddings_path(self) -> Path:
        return self.library_dir / _EMBEDDINGS_FILE

    def _load(self):
        meta_path = self._metadata_path()
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info("Loaded %d model metadata entries", len(self.metadata))

        emb_path = self._embeddings_path()
        if emb_path.exists():
            data = torch.load(emb_path, weights_only=True)
            self.embeddings = {k: v for k, v in data.items()}
            logger.info("Loaded %d model embeddings", len(self.embeddings))

    def _save_metadata(self):
        with open(self._metadata_path(), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _save_embeddings(self):
        torch.save(self.embeddings, self._embeddings_path())

    def register_model(
        self,
        model_id: str,
        file_path: str,
        category: str,
        description: str,
        tags: Optional[list[str]] = None,
        dimensions: Optional[dict] = None,
        offset_y: float = 0.0,
        rotatable: bool = True,
        place_on_floor: bool = True,
        wall_adjacent: bool = False,
    ):
        self.metadata[model_id] = {
            "id": model_id,
            "file_path": file_path,
            "category": category,
            "description": description,
            "tags": tags or [],
            "dimensions": dimensions or {"width": 1.0, "height": 1.0, "depth": 1.0},
            "offset_y": offset_y,
            "rotatable": rotatable,
            "place_on_floor": place_on_floor,
            "wall_adjacent": wall_adjacent,
        }
        desc_texts = [description] + (tags or [])
        emb = encode_text(desc_texts)
        avg_emb = emb.mean(dim=0)
        avg_emb = avg_emb / avg_emb.norm()
        self.embeddings[model_id] = avg_emb
        self._save_metadata()
        self._save_embeddings()
        logger.info("Registered model: %s", model_id)

    def remove_model(self, model_id: str):
        self.metadata.pop(model_id, None)
        self.embeddings.pop(model_id, None)
        self._save_metadata()
        self._save_embeddings()

    def get_model_info(self, model_id: str) -> Optional[dict]:
        return self.metadata.get(model_id)

    def list_models(self) -> list[dict]:
        return list(self.metadata.values())

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.20,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        if not self.embeddings:
            return []

        ids = list(self.embeddings.keys())
        emb_matrix = torch.stack([self.embeddings[mid] for mid in ids])
        similarities = (emb_matrix @ query_embedding.squeeze()).numpy()

        if category_filter:
            filtered = []
            for i, mid in enumerate(ids):
                info = self.metadata.get(mid, {})
                if info.get("category") == category_filter:
                    filtered.append((i, mid, similarities[i]))
        else:
            filtered = [(i, mid, similarities[i]) for i, mid in enumerate(ids)]

        filtered.sort(key=lambda x: x[2], reverse=True)

        results = []
        for _, mid, sim in filtered[:top_k]:
            if sim >= threshold:
                info = self.metadata[mid].copy()
                info["similarity"] = float(sim)
                results.append(info)
        return results

    def build_default_library(self):
        default_models = [
            {
                "model_id": "blue_sofa",
                "file_path": "blue_sofa/scene.gltf",
                "category": "sofa",
                "description": "a blue fabric sofa for living room seating",
                "tags": ["blue sofa", "couch", "seating", "living room", "fabric sofa"],
                "dimensions": {"width": 2.2, "height": 0.9, "depth": 0.9},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "wood_table",
                "file_path": "wood_table/scene.gltf",
                "category": "table",
                "description": "a wooden dining table with natural finish",
                "tags": ["wooden table", "dining table", "wood table", "furniture"],
                "dimensions": {"width": 1.4, "height": 0.75, "depth": 0.8},
                "offset_y": 0.0,
                "wall_adjacent": False,
            },
            {
                "model_id": "armchair",
                "file_path": "armchair/scene.gltf",
                "category": "chair",
                "description": "a comfortable armchair for seating",
                "tags": ["armchair", "chair", "seating", "living room"],
                "dimensions": {"width": 0.9, "height": 0.85, "depth": 0.85},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "bookshelf",
                "file_path": "bookshelf/scene.gltf",
                "category": "shelf",
                "description": "a tall wooden bookshelf for storage",
                "tags": ["bookshelf", "book case", "shelf", "storage", "wooden shelf"],
                "dimensions": {"width": 0.8, "height": 1.8, "depth": 0.35},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "floor_lamp",
                "file_path": "floor_lamp/scene.gltf",
                "category": "lamp",
                "description": "a modern floor lamp for lighting",
                "tags": ["floor lamp", "lamp", "lighting", "standing lamp"],
                "dimensions": {"width": 0.4, "height": 1.6, "depth": 0.4},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "tv_stand",
                "file_path": "tv_stand/scene.gltf",
                "category": "cabinet",
                "description": "a TV stand media cabinet",
                "tags": ["TV stand", "media cabinet", "television stand", "entertainment center"],
                "dimensions": {"width": 1.6, "height": 0.5, "depth": 0.45},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "coffee_table",
                "file_path": "coffee_table/scene.gltf",
                "category": "table",
                "description": "a low coffee table for living room center",
                "tags": ["coffee table", "center table", "low table", "living room table"],
                "dimensions": {"width": 1.2, "height": 0.45, "depth": 0.6},
                "offset_y": 0.0,
                "wall_adjacent": False,
            },
            {
                "model_id": "rug",
                "file_path": "rug/scene.gltf",
                "category": "rug",
                "description": "a decorative area rug for floor",
                "tags": ["rug", "carpet", "area rug", "floor covering"],
                "dimensions": {"width": 2.0, "height": 0.02, "depth": 1.5},
                "offset_y": 0.0,
                "place_on_floor": True,
                "wall_adjacent": False,
            },
            {
                "model_id": "bed",
                "file_path": "bed/scene.gltf",
                "category": "bed",
                "description": "a double bed for bedroom sleeping",
                "tags": ["bed", "double bed", "sleeping", "bedroom", "mattress"],
                "dimensions": {"width": 2.0, "height": 0.5, "depth": 1.6},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "nightstand",
                "file_path": "nightstand/scene.gltf",
                "category": "cabinet",
                "description": "a small nightstand bedside table",
                "tags": ["nightstand", "bedside table", "side table", "bedroom"],
                "dimensions": {"width": 0.5, "height": 0.55, "depth": 0.4},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "desk",
                "file_path": "desk/scene.gltf",
                "category": "table",
                "description": "an office desk for working",
                "tags": ["desk", "office desk", "work table", "study table"],
                "dimensions": {"width": 1.2, "height": 0.75, "depth": 0.6},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "office_chair",
                "file_path": "office_chair/scene.gltf",
                "category": "chair",
                "description": "an office chair with wheels",
                "tags": ["office chair", "desk chair", "swivel chair", "computer chair"],
                "dimensions": {"width": 0.6, "height": 1.2, "depth": 0.6},
                "offset_y": 0.0,
                "wall_adjacent": False,
            },
            {
                "model_id": "wardrobe",
                "file_path": "wardrobe/scene.gltf",
                "category": "cabinet",
                "description": "a tall wardrobe closet for clothes storage",
                "tags": ["wardrobe", "closet", "clothes storage", "armoire"],
                "dimensions": {"width": 1.2, "height": 2.0, "depth": 0.6},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "dining_chair",
                "file_path": "dining_chair/scene.gltf",
                "category": "chair",
                "description": "a wooden dining chair",
                "tags": ["dining chair", "kitchen chair", "wooden chair", "seat"],
                "dimensions": {"width": 0.45, "height": 0.85, "depth": 0.5},
                "offset_y": 0.0,
                "wall_adjacent": False,
            },
            {
                "model_id": "kitchen_cabinet",
                "file_path": "kitchen_cabinet/scene.gltf",
                "category": "cabinet",
                "description": "a kitchen counter cabinet",
                "tags": ["kitchen cabinet", "counter", "kitchen", "cabinet"],
                "dimensions": {"width": 1.8, "height": 0.9, "depth": 0.6},
                "offset_y": 0.0,
                "wall_adjacent": True,
            },
            {
                "model_id": "plant_pot",
                "file_path": "plant_pot/scene.gltf",
                "category": "decor",
                "description": "a potted indoor plant decoration",
                "tags": ["plant", "potted plant", "indoor plant", "decoration", "greenery"],
                "dimensions": {"width": 0.4, "height": 0.8, "depth": 0.4},
                "offset_y": 0.0,
                "wall_adjacent": False,
            },
        ]
        for m in default_models:
            if m["model_id"] not in self.metadata:
                self.register_model(**m)
        logger.info("Default model library built with %d models", len(self.metadata))


_model_library: Optional[ModelLibrary] = None


def get_model_library() -> ModelLibrary:
    global _model_library
    if _model_library is None:
        _model_library = ModelLibrary()
        if not _model_library.metadata:
            _model_library.build_default_library()
    return _model_library
