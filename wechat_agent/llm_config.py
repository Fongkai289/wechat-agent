"""Model profile configuration: defaults, normalization, persistence and key handling."""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

DEFAULT_LLM_CONFIG = {
    "voice": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini-transcribe",
        "api_key": "",
        "api_key_env": "OPENAI_API_KEY",
    },
    "qa": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5-mini",
        "api_key": "",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 1.0,
        "max_context_messages": 60,
        "context_char_budget": 120000,
    },
    "task": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5-mini",
        "api_key": "",
        "api_key_env": "OPENAI_API_KEY",
    },
    "embedding": {
        "enabled": True,
        "base_url": "",
        "model": "text-embedding-3-small",
        "api_key": "",
        "api_key_env": "",
        "dimensions": 0,
        "batch_size": 64,
        "query_candidates": 120,
    },
    "rerank": {
        "enabled": True,
        "base_url": "",
        "model": "gpt-5-nano",
        "api_key": "",
        "api_key_env": "",
        "batch_size": 48,
    },
}


_llm_config_cache: dict[str, tuple[Any, dict[str, Any]]] = {}


_llm_config_cache_lock = threading.Lock()


def load_llm_config(path: Path) -> dict[str, Any]:
    """Return the model configuration, re-reading the file only after it changes."""
    try:
        stat = os.stat(path)
        signature: Any = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    except OSError:
        signature = None
    key = str(path)
    with _llm_config_cache_lock:
        cached = _llm_config_cache.get(key)
    if cached is None or cached[0] != signature:
        cached = (signature, read_llm_config(Path(path)))
        with _llm_config_cache_lock:
            _llm_config_cache[key] = cached
    # Callers mutate profiles freely; never hand out the cached object.
    return copy.deepcopy(cached[1])


def read_llm_config(path: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_LLM_CONFIG))
    raw = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            for name, defaults in DEFAULT_LLM_CONFIG.items():
                incoming = raw.get(name)
                if isinstance(incoming, dict):
                    config[name].update({k: v for k, v in incoming.items() if k in defaults})
    # Legacy tasks used QA settings; snapshot them before independent edits.
    if not isinstance(raw, dict) or not isinstance(raw.get("task"), dict):
        config["task"] = {key: config["qa"][key] for key in DEFAULT_LLM_CONFIG["task"]}
    return normalize_llm_config(config)


def save_llm_config(path: Path, incoming: dict[str, Any]) -> dict[str, Any]:
    current = load_llm_config(path)
    next_config = json.loads(json.dumps(current))
    if isinstance(incoming, dict):
        for name in DEFAULT_LLM_CONFIG:
            profile = incoming.get(name)
            if not isinstance(profile, dict):
                continue
            for key in DEFAULT_LLM_CONFIG[name]:
                if key == "api_key" and profile.get("api_key", "") == "" and current[name].get("api_key"):
                    continue
                if key in profile:
                    next_config[name][key] = profile[key]
            if profile.get("clear_api_key"):
                next_config[name]["api_key"] = ""
    next_config = normalize_llm_config(next_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(next_config, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return next_config


def normalize_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(DEFAULT_LLM_CONFIG))
    for name in DEFAULT_LLM_CONFIG:
        profile = config.get(name) if isinstance(config, dict) else {}
        if not isinstance(profile, dict):
            continue
        default_base_url = out[name]["base_url"]
        base_url = profile.get("base_url")
        if name in {"embedding", "rerank"}:
            out[name]["base_url"] = str(default_base_url if base_url is None else base_url).strip().rstrip("/")
        else:
            out[name]["base_url"] = str(base_url or default_base_url).strip().rstrip("/")
        out[name]["model"] = str(profile.get("model") or out[name]["model"]).strip()
        out[name]["api_key"] = str(profile.get("api_key") or "").strip()
        default_api_key_env = out[name]["api_key_env"]
        api_key_env = profile.get("api_key_env")
        if name in {"embedding", "rerank"}:
            out[name]["api_key_env"] = str(default_api_key_env if api_key_env is None else api_key_env).strip()
        else:
            out[name]["api_key_env"] = str(api_key_env or default_api_key_env).strip()
        if "enabled" in out[name]:
            # Retrieval services are required; migrate old disabled profiles in memory.
            out[name]["enabled"] = True
    try:
        out["qa"]["temperature"] = float(config.get("qa", {}).get("temperature", out["qa"]["temperature"]))
    except (TypeError, ValueError, AttributeError):
        out["qa"]["temperature"] = DEFAULT_LLM_CONFIG["qa"]["temperature"]
    out["qa"]["temperature"] = max(0.0, min(2.0, out["qa"]["temperature"]))
    if uses_default_temperature_only(out["qa"]["model"]):
        out["qa"]["temperature"] = 1.0
    try:
        out["qa"]["max_context_messages"] = int(config.get("qa", {}).get("max_context_messages", out["qa"]["max_context_messages"]))
    except (TypeError, ValueError, AttributeError):
        out["qa"]["max_context_messages"] = DEFAULT_LLM_CONFIG["qa"]["max_context_messages"]
    out["qa"]["max_context_messages"] = max(8, min(160, out["qa"]["max_context_messages"]))
    try:
        out["qa"]["context_char_budget"] = int(config.get("qa", {}).get("context_char_budget", 120000))
    except (TypeError, ValueError, AttributeError):
        out["qa"]["context_char_budget"] = 120000
    out["qa"]["context_char_budget"] = max(10000, min(1000000, out["qa"]["context_char_budget"]))
    embedding_config = config.get("embedding", {}) if isinstance(config, dict) else {}
    try:
        out["embedding"]["dimensions"] = int(embedding_config.get("dimensions", out["embedding"]["dimensions"]))
    except (TypeError, ValueError, AttributeError):
        out["embedding"]["dimensions"] = DEFAULT_LLM_CONFIG["embedding"]["dimensions"]
    out["embedding"]["dimensions"] = max(0, min(3072, out["embedding"]["dimensions"]))
    try:
        out["embedding"]["batch_size"] = int(embedding_config.get("batch_size", out["embedding"]["batch_size"]))
    except (TypeError, ValueError, AttributeError):
        out["embedding"]["batch_size"] = DEFAULT_LLM_CONFIG["embedding"]["batch_size"]
    out["embedding"]["batch_size"] = max(1, min(128, out["embedding"]["batch_size"]))
    try:
        out["embedding"]["query_candidates"] = int(
            embedding_config.get("query_candidates", out["embedding"]["query_candidates"])
        )
    except (TypeError, ValueError, AttributeError):
        out["embedding"]["query_candidates"] = DEFAULT_LLM_CONFIG["embedding"]["query_candidates"]
    out["embedding"]["query_candidates"] = max(10, min(300, out["embedding"]["query_candidates"]))
    rerank_config = config.get("rerank", {}) if isinstance(config, dict) else {}
    try:
        out["rerank"]["batch_size"] = int(rerank_config.get("batch_size", out["rerank"]["batch_size"]))
    except (TypeError, ValueError, AttributeError):
        out["rerank"]["batch_size"] = DEFAULT_LLM_CONFIG["rerank"]["batch_size"]
    out["rerank"]["batch_size"] = max(8, min(120, out["rerank"]["batch_size"]))
    return out


def safe_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    safe = json.loads(json.dumps(config))
    qa_profile = config.get("qa") if isinstance(config.get("qa"), dict) else {}
    for name, profile in safe.items():
        key = str(profile.pop("api_key", "") or "")
        profile["api_key_set"] = bool(key)
        profile["api_key_preview"] = mask_api_key(key)
        profile["env_key_set"] = bool(os.environ.get(profile.get("api_key_env") or ""))
        fallback = qa_profile if name in {"embedding", "rerank"} else None
        profile["effective_api_key_set"] = bool(resolve_llm_api_key(config.get(name) or {}, fallback))
        profile["inherits_qa"] = bool(name in {"embedding", "rerank"} and not key)
    return safe


def mask_api_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "已保存"
    return f"{value[:3]}...{value[-4:]}"


def resolve_llm_api_key(profile: dict[str, Any], fallback: dict[str, Any] | None = None) -> str:
    key = str(profile.get("api_key") or "").strip()
    if key:
        return key
    env_name = str(profile.get("api_key_env") or "").strip()
    if env_name:
        env_key = os.environ.get(env_name, "").strip()
        if env_key:
            return env_key
    if fallback:
        return resolve_llm_api_key(fallback, None)
    return ""


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on", "enabled", "启用"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "停用"}:
        return False
    return default


def effective_llm_profile(config: dict[str, Any], name: str, fallback_name: str = "qa") -> dict[str, Any]:
    profile = dict(config.get(name) or {})
    fallback = config.get(fallback_name) if isinstance(config.get(fallback_name), dict) else {}
    for key in ("base_url", "api_key_env"):
        if not str(profile.get(key) or "").strip() and fallback.get(key):
            profile[key] = fallback.get(key)
    return profile


def uses_default_temperature_only(model: str) -> bool:
    return str(model or "").strip().casefold().startswith("gpt-5")
