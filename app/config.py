from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parent.parent
    model_library_dir: Path = project_root / "data" / "model_library"
    output_dir: Path = project_root / "data" / "output"

    clip_model_name: str = "ViT-B/32"
    clip_device: str = "auto"

    default_room_width: float = 6.0
    default_room_depth: float = 5.0
    default_room_height: float = 3.0

    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.20

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {
        "env_prefix": "TEXT2ROOM_",
        "protected_namespaces": (),
    }


settings = Settings()
