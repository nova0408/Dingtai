from __future__ import annotations

import os
from pathlib import Path


def apply_download_proxy(proxy_url: str) -> None:
    px = str(proxy_url).strip()
    if len(px) == 0:
        return
    os.environ["HTTP_PROXY"] = px
    os.environ["HTTPS_PROXY"] = px
    os.environ["ALL_PROXY"] = px


def prepare_hf_cache_dir(cache_dir: str) -> str:
    path = Path(str(cache_dir).strip())
    path.mkdir(parents=True, exist_ok=True)
    cache_str = str(path)
    os.environ["HF_HOME"] = cache_str
    os.environ["TRANSFORMERS_CACHE"] = str(path / "transformers")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(path / "hub")
    return cache_str


def load_pretrained_with_project_cache(loader, model_id: str, cache_dir: str, local_files_only: bool, role: str):
    store_dir = _project_model_store_dir(cache_dir=cache_dir, role=role, model_id=model_id)
    if store_dir.exists():
        return loader(str(store_dir), local_files_only=True)
    obj = loader(model_id, cache_dir=cache_dir, local_files_only=local_files_only)
    try:
        store_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(obj, "save_pretrained"):
            obj.save_pretrained(str(store_dir))
    except Exception:
        pass
    return obj


def _safe_model_id(model_id: str) -> str:
    return str(model_id).replace("\\", "__").replace("/", "__").replace(":", "__")


def _project_model_store_dir(cache_dir: str, role: str, model_id: str) -> Path:
    return Path(cache_dir) / "project_store" / str(role).strip() / _safe_model_id(model_id)
