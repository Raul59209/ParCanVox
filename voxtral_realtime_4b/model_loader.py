import os
from pathlib import Path
from typing import Any


def _normalize_model_dir(model_dir: str | None) -> str | None:
    if not model_dir:
        return None
    path = Path(model_dir).expanduser()
    return str(path)


def resolve_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def build_hf_kwargs(token: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if token:
        kwargs["token"] = token
    return kwargs


def ensure_hf_token(token: str | None = None) -> str | None:
    resolved = token or resolve_hf_token()
    if resolved:
        os.environ["HF_TOKEN"] = resolved
        os.environ["HUGGING_FACE_HUB_TOKEN"] = resolved
    return resolved


def has_local_model_files(model_dir: str | None) -> bool:
    if not model_dir:
        return False
    path = Path(model_dir).expanduser()
    if not path.exists() or not path.is_dir():
        return False

    expected_files = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
    return any((path / name).exists() for name in expected_files) or any(path.iterdir())


def is_network_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in [
        "network is unreachable",
        "cannot send a request",
        "connection error",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "name or service not known",
        "service unavailable",
    ])


def load_voxtral_components(model_id: str, model_dir: str | None = None, token: str | None = None):
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface/transformers")

    try:
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("transformers does not expose VoxtralRealtimeForConditionalGeneration") from exc

    resolved_dir = _normalize_model_dir(model_dir)
    resolved_token = ensure_hf_token(token)
    kwargs = build_hf_kwargs(resolved_token)

    try:
        import torch
        if resolved_dir and has_local_model_files(resolved_dir):
            processor = AutoProcessor.from_pretrained(resolved_dir, **kwargs)
            model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
                resolved_dir,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                **kwargs,
            )
        else:
            processor = AutoProcessor.from_pretrained(model_id, **kwargs)
            model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                **kwargs,
            )
        return processor, model
    except Exception as exc:
        if is_network_error(exc):
            raise RuntimeError(
                "Unable to load the Voxtral processor/model because Hugging Face could not be reached. "
                "Set HF_TOKEN if your instance requires authentication, or mount a local model directory "
                "via VOXTRAL_MODEL_DIR and rerun the container."
            ) from exc
        raise