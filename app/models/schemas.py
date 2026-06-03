from pydantic import BaseModel, Field
from typing import Optional


class SceneRequest(BaseModel):
    text: str = Field(..., description="自然语言描述，如'一个带有蓝色沙发和木桌的客厅'")
    room_width: Optional[float] = Field(None, gt=0, description="房间宽度(米)")
    room_depth: Optional[float] = Field(None, gt=0, description="房间深度(米)")
    room_height: Optional[float] = Field(None, gt=0, description="房间高度(米)")
    top_k: int = Field(5, ge=1, le=20, description="检索返回的top-k模型数")
    similarity_threshold: Optional[float] = Field(None, ge=0, le=1, description="相似度阈值")
    seed: Optional[int] = Field(None, description="随机种子，用于可复现布局")
    render: bool = Field(True, description="是否生成渲染预览图")
    render_width: int = Field(1280, ge=320, le=3840, description="渲染图宽度(像素)")
    render_height: int = Field(720, ge=240, le=2160, description="渲染图高度(像素)")
    render_views: int = Field(4, ge=1, le=8, description="渲染视角数量")


class ModelRegistration(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str = Field(..., description="模型唯一ID")
    file_path: str = Field(..., description="GLTF模型文件路径")
    category: str = Field(..., description="模型分类(sofa/table/chair/lamp...)")
    description: str = Field(..., description="英文描述，用于CLIP匹配")
    tags: list[str] = Field(default_factory=list, description="额外标签")
    dimensions: dict = Field(
        default_factory=lambda: {"width": 1.0, "height": 1.0, "depth": 1.0},
        description="模型尺寸(米)",
    )
    offset_y: float = Field(0.0, description="Y轴偏移")
    rotatable: bool = Field(True, description="是否可旋转")
    place_on_floor: bool = Field(True, description="是否放置在地板上")
    wall_adjacent: bool = Field(False, description="是否靠墙放置")


class SceneResponse(BaseModel):
    status: str
    text: str
    room_type: str
    matched_models: list[dict]
    layout: Optional[dict] = None
    output_path: Optional[str] = None
    render_paths: list[str] = []


class ModelInfo(BaseModel):
    id: str
    category: str
    description: str
    tags: list[str]
    dimensions: dict
    similarity: Optional[float] = None
