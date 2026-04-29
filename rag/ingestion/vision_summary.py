from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import requests

from rag.config import Settings


VISION_PROMPT = (
    "Describe this engineering image for retrieval. Focus on components, flows, labels, "
    "metrics, and operational meaning. Keep it to 1-2 concise sentences."
)


@dataclass(slots=True)
class VisionSummary:
    summary: str = ""
    engine: str = "none"
    status: str = "missing"


def generate_image_summary(
    asset_path: Path,
    settings: Settings,
) -> VisionSummary:
    if not settings.image_vision_summary_enabled:
        return VisionSummary(status="disabled")

    cached = _read_cache(_cache_path(asset_path, settings.asset_cache_dir, suffix="vision"), asset_path)
    if cached is not None and cached.status not in {"unconfigured", "unavailable", "error", "missing"}:
        return cached

    summary = _run_vision_summary(asset_path=asset_path, settings=settings)
    _write_cache(
        _cache_path(asset_path, settings.asset_cache_dir, suffix="vision"),
        asset_path=asset_path,
        payload={
            "summary": summary.summary,
            "engine": summary.engine,
            "status": summary.status,
        },
    )
    return summary


def _run_vision_summary(asset_path: Path, settings: Settings) -> VisionSummary:
    if not asset_path.exists():
        return VisionSummary(status="missing")
    if not settings.ollama_vision_model:
        return VisionSummary(status="unconfigured")

    try:
        prepared_path = _prepare_asset_for_vision(asset_path)
        encoded_image = base64.b64encode(prepared_path.read_bytes()).decode("utf-8")
        response = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/chat",
            json={
                "model": settings.ollama_vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": VISION_PROMPT,
                        "images": [encoded_image],
                    }
                ],
                "stream": False,
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message", {}) or {}
        content = str(message.get("content", "")).strip()
        cleaned = " ".join(content.split()).strip()
        return VisionSummary(
            summary=cleaned,
            engine=f"ollama:{settings.ollama_vision_model}",
            status="ok" if cleaned else "empty",
        )
    except (OSError, subprocess.CalledProcessError, requests.RequestException, ValueError, KeyError):
        return VisionSummary(
            engine=f"ollama:{settings.ollama_vision_model}",
            status="error",
        )
    finally:
        if "prepared_path" in locals() and prepared_path != asset_path and prepared_path.exists():
            temp_parent = prepared_path.parent
            prepared_path.unlink(missing_ok=True)
            temp_parent.rmdir()


def _cache_path(asset_path: Path, cache_dir: Path, suffix: str) -> Path:
    relative = asset_path.name + f".{suffix}.json"
    hashed_parent = sha256(str(asset_path.parent).encode("utf-8")).hexdigest()[:12]
    return cache_dir / hashed_parent / relative


def _read_cache(cache_path: Path, asset_path: Path) -> VisionSummary | None:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    current_hash = sha256(asset_path.read_bytes()).hexdigest() if asset_path.exists() else ""
    if payload.get("asset_sha256") != current_hash:
        return None
    return VisionSummary(
        summary=str(payload.get("summary", "")).strip(),
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


def _prepare_asset_for_vision(asset_path: Path) -> Path:
    if asset_path.suffix.lower() != ".svg":
        return asset_path

    if shutil.which("sips") is None:
        raise OSError("sips is not available for SVG rasterization")

    temp_dir = Path(tempfile.mkdtemp(prefix="speclens-vision-"))
    output_path = temp_dir / f"{asset_path.stem}.png"
    subprocess.run(
        [
            "sips",
            "-s",
            "format",
            "png",
            str(asset_path),
            "--out",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path
