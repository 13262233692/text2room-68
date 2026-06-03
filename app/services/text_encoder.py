import logging
from typing import Union

import torch
import clip

from app.config import settings

logger = logging.getLogger(__name__)

_device = None
_model = None
_preprocess = None


def _get_device() -> torch.device:
    if settings.clip_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(settings.clip_device)


def get_clip_model():
    global _model, _preprocess, _device
    if _model is None:
        _device = _get_device()
        logger.info("Loading CLIP model %s on %s", settings.clip_model_name, _device)
        _model, _preprocess = clip.load(settings.clip_model_name, device=_device)
        _model.eval()
        logger.info("CLIP model loaded")
    return _model, _preprocess, _device


@torch.no_grad()
def encode_text(text: Union[str, list[str]]) -> torch.Tensor:
    model, _, device = get_clip_model()
    if isinstance(text, str):
        text = [text]
    tokens = clip.tokenize(text, truncate=True).to(device)
    features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu()


@torch.no_grad()
def encode_images(images) -> torch.Tensor:
    model, preprocess, device = get_clip_model()
    from PIL import Image
    import numpy as np

    processed = []
    for img in images:
        if isinstance(img, (str,)):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        processed.append(preprocess(img))

    batch = torch.stack(processed).to(device)
    features = model.encode_image(batch)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu()
