"""Phase 0 data preparation for Tiki fashion RAG pipeline."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BOILERPLATE_THRESHOLD = 0.05

COLOR_KEYWORDS = re.compile(
    r"(mau|color|bang\s*mau|chon\s*mau|nhom\s*mau|mau\s*chu\s*dao|mau\s*ao|mau\s*sac|mau\s*sac)",
    re.IGNORECASE,
)
SIZE_KEYWORDS = re.compile(
    r"(size|kich\s*(co|thuoc)|bang\s*size|chon\s*size|co\s*ao|lua\s*size)",
    re.IGNORECASE,
)

COUNTRY_ALIASES: dict[str, str] = {
    "viet nam": "Việt Nam",
    "viet_nam": "Việt Nam",
    "han quoc": "Hàn Quốc",
    "han_quoc": "Hàn Quốc",
    "trung quoc": "Trung Quốc",
    "trung_quoc": "Trung Quốc",
    "nhat ban": "Nhật Bản",
    "nhat_ban": "Nhật Bản",
    "thai lan": "Thái Lan",
    "thai_lan": "Thái Lan",
    "an do": "Ấn Độ",
    "an_do": "Ấn Độ",
    "indonesia": "Indonesia",
    "malaysia": "Malaysia",
    "usa": "Mỹ",
    "my": "Mỹ",
    "phap": "Pháp",
    "duc": "Đức",
    "italia": "Ý",
    "y": "Ý",
}

COLOR_VALUE_MAP: dict[str, str] = {
    "den": "đen",
    "trang": "trắng",
    "do": "đỏ",
    "xanh": "xanh",
    "xanh la": "xanh lá",
    "xanh duong": "xanh dương",
    "xanh navy": "xanh navy",
    "xanh mint": "xanh mint",
    "xanh den": "xanh đen",
    "hong": "hồng",
    "vang": "vàng",
    "cam": "cam",
    "tim": "tím",
    "tim than": "tím than",
    "nau": "nâu",
    "be": "be",
    "kem": "kem",
    "ghi": "ghi",
    "bac": "bạc",
    "do do": "đỏ đô",
}

SIZE_VALUE_PATTERN = re.compile(
    r"^(xxs|xs|s|m|l|xl|xxl|xxxl|2xl|3xl|4xl|5xl|freesize|free\s*size|one\s*size|os)$",
    re.IGNORECASE,
)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_key(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = strip_accents(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_image_hash(url: str) -> str | None:
    path = urlparse(url).path
    match = re.search(r"/([a-f0-9]{32})\.(jpg|jpeg|png|webp)$", path, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return Path(path).stem.lower() or None


def dedup_image_urls(image_urls: list[str]) -> list[str]:
    """Dedup by image hash, prefer w1200 URLs."""
    by_hash: dict[str, tuple[int, int, str]] = {}
    for order, url in enumerate(image_urls):
        image_hash = extract_image_hash(url)
        if not image_hash:
            continue
        priority = 0 if "/cache/w1200/" in url else 1
        current = by_hash.get(image_hash)
        if current is None or priority < current[0] or (priority == current[0] and order < current[1]):
            by_hash[image_hash] = (priority, order, url)
    return [url for _, _, url in sorted(by_hash.values(), key=lambda item: item[1])]


def dedup_products(raw_products: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove exact duplicate product_id rows and merge near-duplicate listings."""
    stats = {"removed_exact_dup": 0, "merged_title_dup": 0}
    seen_ids: set[int] = set()
    deduped: list[dict[str, Any]] = []

    for product in raw_products:
        product_id = int(product["product_id"])
        if product_id in seen_ids:
            stats["removed_exact_dup"] += 1
            continue
        seen_ids.add(product_id)
        deduped.append(product)

    merged: list[dict[str, Any]] = []
    index_by_signature: dict[tuple[str, str], int] = {}
    for product in deduped:
        signature = (product.get("title", ""), product.get("description", ""))
        if signature in index_by_signature:
            existing_idx = index_by_signature[signature]
            existing = merged[existing_idx]
            existing_urls = dedup_image_urls(
                (existing.get("image_urls") or []) + (product.get("image_urls") or [])
            )
            existing["image_urls"] = existing_urls
            stats["merged_title_dup"] += 1
            continue
        index_by_signature[signature] = len(merged)
        merged.append(product)

    return merged, stats


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    # Insert boundaries before common Tiki boilerplate markers in run-on descriptions.
    text = re.sub(
        r"(?<=\s)(Giá\s*sản_phẩm|Giá\s*sản\s*phẩm|HOA\s*ĐƠN\s*VAT|HÓA\s*ĐƠN\s*VAT)",
        r". \1",
        text,
        flags=re.IGNORECASE,
    )
    chunks = re.split(r"(?<=[.!?…])\s+|\s*\.\s*\.\s*\.?\s*", text)
    sentences = [chunk.strip(" .") for chunk in chunks if chunk and chunk.strip(" .")]
    return sentences


def strip_known_boilerplate_substrings(description: str, boilerplate: set[str]) -> str:
    """Remove boilerplate phrases even when embedded in long run-on text."""
    cleaned = description
    for phrase in sorted(boilerplate, key=len, reverse=True):
        pattern = re.escape(phrase).replace(r"\ ", r"\s+")
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned


def normalize_sentence(sentence: str) -> str:
    sentence = sentence.lower().strip()
    sentence = sentence.replace("_", " ")
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"[^\w\s]", "", sentence, flags=re.UNICODE)
    return strip_accents(sentence)


def detect_boilerplate_sentences(
    products: list[dict[str, Any]],
    threshold: float = BOILERPLATE_THRESHOLD,
) -> set[str]:
    counter: Counter[str] = Counter()
    for product in products:
        description = product.get("description") or ""
        seen_in_product: set[str] = set()
        for sentence in split_sentences(description):
            normalized = normalize_sentence(sentence)
            if len(normalized) < 20:
                continue
            seen_in_product.add(normalized)
        counter.update(seen_in_product)

    min_count = max(1, int(len(products) * threshold))
    return {sentence for sentence, count in counter.items() if count >= min_count}


def remove_boilerplate(description: str, boilerplate: set[str]) -> str:
    kept: list[str] = []
    for sentence in split_sentences(description):
        if normalize_sentence(sentence) in boilerplate:
            continue
        kept.append(sentence.strip())
    merged = " ".join(kept).strip()
    return strip_known_boilerplate_substrings(merged, boilerplate)


def classify_option_name(option_name: str) -> str:
    normalized = normalize_key(option_name)
    if COLOR_KEYWORDS.search(normalized):
        return "color"
    if SIZE_KEYWORDS.search(normalized):
        return "size"
    return "variant"


def normalize_color_value(raw_value: str) -> str:
    key = normalize_key(raw_value)
    if key in COLOR_VALUE_MAP:
        return COLOR_VALUE_MAP[key]
    for alias, canonical in COLOR_VALUE_MAP.items():
        if alias in key:
            return canonical
    return raw_value.strip()


def normalize_size_value(raw_value: str) -> str:
    value = raw_value.strip()
    upper = re.sub(r"\s+", "", value.upper())
    # Extract leading size token e.g. "L (55-59kg)" -> "L"
    match = re.match(r"^([A-Z0-9]+)", upper)
    if match and SIZE_VALUE_PATTERN.match(match.group(1)):
        return match.group(1)
    if SIZE_VALUE_PATTERN.match(upper):
        return upper.replace(" ", "")
    return value


def normalize_configurable_options(options: list[dict[str, Any]] | None) -> dict[str, list[dict[str, str]]]:
    attrs: dict[str, list[dict[str, str]]] = {"color": [], "size": [], "variant": []}
    if not options:
        return attrs

    for option in options:
        canonical = classify_option_name(option.get("name", ""))
        for value in option.get("values") or []:
            raw_value = str(value.get("label", "")).strip()
            if not raw_value:
                continue
            if canonical == "color":
                norm_value = normalize_color_value(raw_value)
            elif canonical == "size":
                norm_value = normalize_size_value(raw_value)
            else:
                norm_value = raw_value
            attrs[canonical].append({"raw_value": raw_value, "value": norm_value})
    return attrs


def normalize_origin(origin: str | None) -> list[str]:
    if not origin or not str(origin).strip():
        return ["unknown"]

    text = str(origin).replace("_", " ")
    text = re.sub(r"\s*\.\.\.\s*", " ", text)
    parts = re.split(r"[,;/|]", text)
    countries: list[str] = []
    for part in parts:
        key = normalize_key(part)
        if not key or key in {"...", "unknown"}:
            continue
        mapped = COUNTRY_ALIASES.get(key)
        if mapped:
            countries.append(mapped)
            continue
        for alias, canonical in COUNTRY_ALIASES.items():
            if alias in key:
                countries.append(canonical)
                break
        else:
            display = part.strip().replace("_", " ")
            if display:
                countries.append(display)

    return list(dict.fromkeys(countries)) or ["unknown"]


def get_category_l2(breadcrumbs: list[str] | None) -> str:
    if not breadcrumbs:
        return ""
    if len(breadcrumbs) >= 3:
        return breadcrumbs[2].replace("_", " ")
    return breadcrumbs[-1].replace("_", " ")


def build_text_fields(
    title: str,
    breadcrumbs: list[str] | None,
    brand: str,
    description: str,
) -> tuple[str, str]:
    category_parts = [part.replace("_", " ") for part in (breadcrumbs or [])[:-1]]
    category = " > ".join(category_parts)
    parts = [title, category, brand, description]
    text_segmented = " | ".join(part for part in parts if part)
    text_natural = text_segmented.replace("_", " ")
    return text_segmented, text_natural


def clean_product(
    product: dict[str, Any],
    boilerplate: set[str],
) -> dict[str, Any]:
    title = (product.get("title") or "").strip()
    brand = ((product.get("brand") or {}).get("name") or "").strip()
    description_raw = product.get("description") or ""
    description_clean = remove_boilerplate(description_raw, boilerplate)

    text_segmented, text_natural = build_text_fields(
        title=title,
        breadcrumbs=product.get("breadcrumbs"),
        brand=brand,
        description=description_clean,
    )

    return {
        "product_id": int(product["product_id"]),
        "title": title.replace("_", " "),
        "text_segmented": text_segmented,
        "text_natural": text_natural,
        "description_clean": description_clean,
        "category_l2": get_category_l2(product.get("breadcrumbs")),
        "breadcrumbs": [part.replace("_", " ") for part in (product.get("breadcrumbs") or [])],
        "brand": brand,
        "origin_norm": normalize_origin(product.get("origin")),
        "attrs": normalize_configurable_options(product.get("configurable_options")),
        "image_urls_unique": dedup_image_urls(product.get("image_urls") or []),
    }


def preprocess_products(
    raw_products: list[dict[str, Any]],
    boilerplate_threshold: float = BOILERPLATE_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deduped, dedup_stats = dedup_products(raw_products)
    boilerplate = detect_boilerplate_sentences(deduped, threshold=boilerplate_threshold)
    cleaned = [clean_product(product, boilerplate) for product in deduped]

    stats = {
        **dedup_stats,
        "input_count": len(raw_products),
        "output_count": len(cleaned),
        "boilerplate_sentence_count": len(boilerplate),
        "boilerplate_threshold": boilerplate_threshold,
        "raw_image_urls": sum(len(p.get("image_urls") or []) for p in raw_products),
        "unique_image_urls": sum(len(p["image_urls_unique"]) for p in cleaned),
    }
    metadata = {"stats": stats, "boilerplate_sentences": sorted(boilerplate)}
    return cleaned, metadata


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_preprocess_pipeline(
    input_path: Path,
    output_dir: Path,
    boilerplate_threshold: float = BOILERPLATE_THRESHOLD,
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as file:
        raw_products = json.load(file)

    cleaned, metadata = preprocess_products(raw_products, boilerplate_threshold=boilerplate_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(cleaned, output_dir / "products_clean.jsonl")
    with (output_dir / "preprocess_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    with (output_dir / "boilerplate_sentences.json").open("w", encoding="utf-8") as file:
        json.dump(metadata["boilerplate_sentences"], file, ensure_ascii=False, indent=2)

    return metadata["stats"]
