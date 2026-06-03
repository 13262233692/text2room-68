import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import (
    SceneRequest,
    SceneResponse,
    ModelRegistration,
    ModelInfo,
)
from app.services.pipeline import get_pipeline
from app.services.model_library import get_model_library

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Initializing Text2Room service...")
    get_pipeline()
    logger.info("Text2Room service ready")
    yield
    logger.info("Text2Room service shutting down")


app = FastAPI(
    title="Text2Room - Text to 3D Indoor Scene Generator",
    description="输入自然语言描述，自动检索3D模型并生成GLTF室内场景",
    version="1.0.0",
    lifespan=lifespan,
)

output_dir = settings.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/scenes", StaticFiles(directory=str(output_dir)), name="scenes")


@app.post("/api/generate", response_model=SceneResponse)
async def generate_scene(request: SceneRequest):
    pipeline = get_pipeline()
    result = pipeline.run(
        text=request.text,
        room_width=request.room_width,
        room_depth=request.room_depth,
        room_height=request.room_height,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        seed=request.seed,
        render=request.render,
        render_width=request.render_width,
        render_height=request.render_height,
        render_views=request.render_views,
    )
    return SceneResponse(**result)


@app.get("/api/models", response_model=list[ModelInfo])
async def list_models():
    library = get_model_library()
    models = library.list_models()
    return [ModelInfo(**m) for m in models]


@app.post("/api/models", response_model=ModelInfo)
async def register_model(reg: ModelRegistration):
    library = get_model_library()
    if library.get_model_info(reg.model_id):
        raise HTTPException(status_code=409, detail=f"Model '{reg.model_id}' already exists")
    library.register_model(
        model_id=reg.model_id,
        file_path=reg.file_path,
        category=reg.category,
        description=reg.description,
        tags=reg.tags,
        dimensions=reg.dimensions,
        offset_y=reg.offset_y,
        rotatable=reg.rotatable,
        place_on_floor=reg.place_on_floor,
        wall_adjacent=reg.wall_adjacent,
    )
    info = library.get_model_info(reg.model_id)
    return ModelInfo(**info)


@app.delete("/api/models/{model_id}")
async def remove_model(model_id: str):
    library = get_model_library()
    if not library.get_model_info(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    library.remove_model(model_id)
    return {"status": "removed", "model_id": model_id}


@app.post("/api/models/rebuild")
async def rebuild_library():
    library = get_model_library()
    library.build_default_library()
    return {"status": "rebuilt", "count": len(library.metadata)}


@app.get("/api/scene/{scene_id}/download")
async def download_scene(scene_id: str):
    scene_dir = settings.output_dir / scene_id
    gltf_path = scene_dir / "scene.gltf"
    if not gltf_path.exists():
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found")
    return FileResponse(
        path=str(gltf_path),
        media_type="model/gltf+json",
        filename=f"{scene_id}.gltf",
    )


@app.get("/api/scene/{scene_id}/render")
async def list_renders(scene_id: str):
    renders_dir = settings.output_dir / scene_id / "renders"
    if not renders_dir.exists():
        raise HTTPException(status_code=404, detail=f"No renders for scene '{scene_id}'")
    images = sorted(renders_dir.glob("view_*.png"))
    if not images:
        raise HTTPException(status_code=404, detail=f"No renders for scene '{scene_id}'")
    return {
        "scene_id": scene_id,
        "renders": [f"/scenes/{scene_id}/renders/{img.name}" for img in images],
    }


@app.get("/api/scene/{scene_id}/render/{view_name}")
async def get_render(scene_id: str, view_name: str):
    render_path = settings.output_dir / scene_id / "renders" / view_name
    if not render_path.exists():
        raise HTTPException(status_code=404, detail=f"Render '{view_name}' not found")
    return FileResponse(
        path=str(render_path),
        media_type="image/png",
        filename=view_name,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "text2room"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
