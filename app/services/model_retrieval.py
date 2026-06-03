import logging
import re
from typing import Optional

import torch

from app.config import settings
from app.services.model_library import ModelLibrary
from app.services.text_encoder import encode_text

logger = logging.getLogger(__name__)

ROOM_TYPE_KEYWORDS = {
    "living_room": ["客厅", "起居室", "living room", "lounge", "sitting room"],
    "bedroom": ["卧室", "睡房", "bedroom", "bed room"],
    "office": ["办公室", "书房", "office", "study", "workroom"],
    "kitchen": ["厨房", "kitchen", "cooking room"],
    "dining_room": ["餐厅", "饭厅", "dining room", "dining"],
}

FURNITURE_KEYWORDS = {
    "sofa": ["沙发", "sofa", "couch"],
    "chair": ["椅子", "座椅", "chair", "seat", "armchair", "办公椅", "dining chair"],
    "table": ["桌子", "桌", "table", "desk"],
    "bed": ["床", "bed"],
    "cabinet": ["柜", "cabinet", "shelf", "书架", "衣柜", "wardrobe", "bookshelf"],
    "lamp": ["灯", "lamp", "light"],
    "rug": ["地毯", "rug", "carpet"],
    "decor": ["装饰", "植物", "plant", "decoration", "花盆"],
}

CATEGORY_MAP = {
    "sofa": "sofa",
    "chair": "chair",
    "table": "table",
    "bed": "bed",
    "cabinet": "cabinet",
    "lamp": "lamp",
    "rug": "rug",
    "decor": "decor",
}


def detect_room_type(text: str) -> str:
    text_lower = text.lower()
    for room_type, keywords in ROOM_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return room_type
    return "living_room"


def _extract_quantity(text: str, keyword: str) -> int:
    cn_nums = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    en_nums = {
        "a ": 1, "an ": 1,
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a pair of": 2, "couple of": 2, "few": 3, "several": 4,
    }
    classifiers = ["把", "张", "个", "盏", "件", "套", "台", "把", "个"]

    text_lower = text.lower()
    kw_pos = text_lower.find(keyword)
    if kw_pos <= 0:
        return 1

    prefix = text_lower[:kw_pos]
    matches = []

    for num_phrase, num in sorted(en_nums.items(), key=lambda x: -len(x[0])):
        start = 0
        while True:
            pos = prefix.find(num_phrase, start)
            if pos == -1:
                break
            dist = kw_pos - (pos + len(num_phrase))
            if dist >= 0:
                matches.append((dist, num))
            start = pos + 1

    for ch, num in cn_nums.items():
        start = 0
        while True:
            pos = prefix.find(ch, start)
            if pos == -1:
                break
            if pos + 1 < len(prefix) and prefix[pos + 1] in classifiers:
                dist = kw_pos - (pos + 2)
                if dist >= 0:
                    matches.append((dist, num))
            start = pos + 1

    digit_matches = re.finditer(r"(\d+)", prefix)
    for m in digit_matches:
        num = int(m.group(1))
        if 1 <= num <= 20:
            dist = kw_pos - m.end()
            if dist >= 0:
                matches.append((dist, num))

    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    return 1


def extract_furniture_requests(text: str) -> list[dict]:
    text_lower = text.lower()
    requests = []
    seen = set()

    for furniture_type, keywords in FURNITURE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower and furniture_type not in seen:
                seen.add(furniture_type)
                color = _extract_color(text_lower, kw)
                quantity = _extract_quantity(text_lower, kw)
                requests.append({
                    "type": furniture_type,
                    "color": color,
                    "quantity": quantity,
                    "original_keyword": kw,
                    "search_query": f"{color} {furniture_type}" if color else furniture_type,
                })
                break

    if not requests:
        requests.append({
            "type": "generic",
            "color": None,
            "quantity": 1,
            "original_keyword": text,
            "search_query": text,
        })

    return requests


def _extract_color(text: str, keyword: str) -> Optional[str]:
    colors = [
        "blue", "蓝", "red", "红", "green", "绿", "yellow", "黄",
        "white", "白", "black", "黑", "brown", "棕", "木色",
        "gray", "灰", "grey", "orange", "橙", "pink", "粉",
        "purple", "紫", "beige", "米色", "wooden", "木制", "木",
    ]
    pattern = re.compile(r"|".join(colors))
    kw_pos = text.find(keyword)
    if kw_pos > 0:
        prefix = text[:kw_pos]
        match = pattern.search(prefix)
        if match:
            return match.group()
    return None


def retrieve_models(
    text: str,
    library: ModelLibrary,
    top_k: int = 5,
    threshold: Optional[float] = None,
) -> list[dict]:
    threshold = threshold or settings.retrieval_similarity_threshold
    furniture_requests = extract_furniture_requests(text)
    all_results = []

    for req in furniture_requests:
        query = req["search_query"]
        quantity = req.get("quantity", 1)
        logger.info("Searching for: '%s' (type=%s, quantity=%d)", query, req["type"], quantity)

        query_emb = encode_text(query)
        results = library.search(
            query_embedding=query_emb,
            top_k=top_k,
            threshold=threshold,
            category_filter=CATEGORY_MAP.get(req["type"]),
        )

        if not results:
            query_emb = encode_text(query)
            results = library.search(
                query_embedding=query_emb,
                top_k=top_k,
                threshold=threshold * 0.5,
            )

        if results:
            best = results[0]
            best["requested_type"] = req["type"]
            best["requested_color"] = req.get("color")
            best["requested_quantity"] = quantity
            best["original_model_id"] = best["id"]
            best["quantity"] = quantity

            if quantity <= 1:
                best["instance_id"] = f"{best['id']}_0"
                all_results.append(best)
                logger.info("Matched: %s (sim=%.3f, qty=%d)", best["id"], best["similarity"], quantity)
            else:
                for i in range(quantity):
                    instance = best.copy()
                    instance["instance_id"] = f"{best['id']}_{i}"
                    instance["id"] = f"{best['id']}_{i}"
                    instance["quantity_index"] = i
                    instance["quantity_total"] = quantity
                    all_results.append(instance)
                logger.info("Matched: %s (sim=%.3f, qty=%d instances)", best["original_model_id"], best["similarity"], quantity)
        else:
            logger.warning("No match found for: %s", query)

    return all_results
