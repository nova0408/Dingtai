from __future__ import annotations

import os
from pathlib import Path

from loguru import logger


# region HuggingFace 缓存
def apply_download_proxy(proxy_url: str) -> None:
    """向环境变量注入模型下载代理。

    Parameters
    ----------
    proxy_url:
        代理地址；空字符串表示不注入。
    """
    px = str(proxy_url).strip()
    if len(px) == 0:
        logger.info("未注入下载代理（proxy_url 为空）")
        return
    os.environ["HTTP_PROXY"] = px
    os.environ["HTTPS_PROXY"] = px
    os.environ["ALL_PROXY"] = px
    logger.info(f"已注入下载代理：{px}")


def prepare_hf_cache_dir(cache_dir: str) -> str:
    """准备 HuggingFace 项目缓存目录。

    Parameters
    ----------
    cache_dir:
        缓存根目录路径。

    Returns
    -------
    cache_str:
        规范化后的缓存目录字符串，同时写入 HF 相关环境变量。
    """
    path = Path(str(cache_dir).strip())
    path.mkdir(parents=True, exist_ok=True)
    cache_str = str(path)
    os.environ["HF_HOME"] = cache_str
    os.environ["TRANSFORMERS_CACHE"] = str(path / "transformers")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(path / "hub")
    return cache_str


def load_pretrained_with_project_cache(loader, model_id: str, cache_dir: str, local_files_only: bool, role: str):
    """从项目缓存或 HuggingFace 缓存加载模型/processor。

    Parameters
    ----------
    loader:
        `from_pretrained` 函数。
    model_id:
        HuggingFace 模型 ID 或本地路径。
    cache_dir:
        HuggingFace 缓存根目录。
    local_files_only:
        是否只使用本地缓存。
    role:
        缓存角色名，用于区分 processor/model。

    Returns
    -------
    obj:
        加载后的模型或 processor 对象。
    """
    store_dir = _project_model_store_dir(cache_dir=cache_dir, role=role, model_id=model_id)
    if store_dir.exists():
        return loader(str(store_dir), local_files_only=True)
    try:
        obj = loader(model_id, cache_dir=cache_dir, local_files_only=local_files_only)
    except Exception as exc:
        if not local_files_only:
            raise
        logger.warning(f"{role} 项目缓存未命中，尝试从全局 HuggingFace 缓存迁移：{model_id}")
        try:
            obj = loader(model_id, local_files_only=True)
        except Exception as inner_exc:
            raise RuntimeError(f"{role} 本地缓存不可用：{model_id}") from inner_exc
    try:
        store_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(obj, "save_pretrained"):
            obj.save_pretrained(str(store_dir))
            logger.info(f"{role} 已写入项目缓存：{store_dir}")
    except Exception as save_exc:
        logger.warning(f"{role} 写入项目缓存失败：{save_exc}")
    return obj


def _safe_model_id(model_id: str) -> str:
    """把模型 ID 转成可用于本地目录名的字符串。"""
    return str(model_id).replace("\\", "__").replace("/", "__").replace(":", "__")


def _project_model_store_dir(cache_dir: str, role: str, model_id: str) -> Path:
    """生成项目内模型副本目录。"""
    return Path(cache_dir) / "project_store" / str(role).strip() / _safe_model_id(model_id)


# endregion
