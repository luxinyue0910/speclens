from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from rag.config import Settings


SVG_EXTENSIONS = {".svg"}
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class OCRExtraction:
    text: str = ""
    engine: str = "none"
    status: str = "missing"


def extract_image_text(asset_path: Path, settings: Settings) -> OCRExtraction:
    if not settings.image_ocr_enabled:
        return OCRExtraction(status="disabled")

    cached = _read_cache(_cache_path(asset_path, settings.asset_cache_dir, suffix="ocr"), asset_path)
    if cached is not None and cached.status not in {"unavailable", "error", "missing"}:
        return cached

    extraction = _run_ocr(asset_path=asset_path, settings=settings)
    _write_cache(
        _cache_path(asset_path, settings.asset_cache_dir, suffix="ocr"),
        asset_path=asset_path,
        payload={
            "text": extraction.text,
            "engine": extraction.engine,
            "status": extraction.status,
        },
    )
    return extraction


def _run_ocr(asset_path: Path, settings: Settings) -> OCRExtraction:
    if not asset_path.exists():
        return OCRExtraction(status="missing")

    suffix = asset_path.suffix.lower()
    if suffix in SVG_EXTENSIONS:
        text = _extract_svg_text(asset_path)
        return OCRExtraction(
            text=text,
            engine="svg_text" if text else "svg_text",
            status="ok" if text else "empty",
        )

    if suffix not in RASTER_EXTENSIONS:
        return OCRExtraction(status="unsupported")

    if shutil.which(settings.tesseract_cmd) is None:
        return OCRExtraction(status="unavailable")

    try:
        completed = subprocess.run(
            [
                settings.tesseract_cmd,
                str(asset_path),
                "stdout",
                "-l",
                settings.ocr_language,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return OCRExtraction(engine="tesseract", status="error")

    text = " ".join(completed.stdout.split()).strip()
    return OCRExtraction(
        text=text,
        engine="tesseract",
        status="ok" if text else "empty",
    )


def _extract_svg_text(asset_path: Path) -> str:
    try:
        tree = ET.parse(asset_path)
    except ET.ParseError:
        return ""

    texts: list[str] = []
    for element in tree.iter():
        tag = element.tag.rsplit("}", maxsplit=1)[-1]
        if tag not in {"text", "tspan"}:
            continue
        value = " ".join("".join(element.itertext()).split()).strip()
        if value:
            texts.append(value)
    return " ".join(texts)


def _cache_path(asset_path: Path, cache_dir: Path, suffix: str) -> Path:
    relative = asset_path.name + f".{suffix}.json"
    hashed_parent = sha256(str(asset_path.parent).encode("utf-8")).hexdigest()[:12]
    return cache_dir / hashed_parent / relative


def _read_cache(cache_path: Path, asset_path: Path) -> OCRExtraction | None:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    current_hash = sha256(asset_path.read_bytes()).hexdigest() if asset_path.exists() else ""
    if payload.get("asset_sha256") != current_hash:
        return None
    return OCRExtraction(
        text=str(payload.get("text", "")).strip(),
        engine=str(payload.get("engine", "none")).strip() or "none",
        status=str(payload.get("status", "missing")).strip() or "missing",
    )


def _write_cache(cache_path: Path, asset_path: Path, payload: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        **payload,
        "asset_sha256": sha256(asset_path.read_bytes()).hexdigest() if asset_path.exists() else "",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    cache_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
