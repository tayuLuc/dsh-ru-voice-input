"""GigaAM v3 sidecar for the DSH speech provider.

One process, two Python backends, chosen by ``GIGAAM_ENGINE``:

``onnx-asr``
    The portable default. ``onnx_asr`` runs on CPU everywhere and on CUDA where
    present, and the ``gigaam-v3-e2e-*`` heads emit punctuation and
    capitalisation. This is the engine for a host that is not Apple Silicon,
    and the one to keep if only a single engine can be installed.

``mlx``
    The Apple Silicon path. ``gigaam_mlx`` is the MLX port of the same e2e
    models, noticeably faster than ONNX on an M-series GPU, and also punctuates.

Both are pure Python wheels with no PyTorch, so one virtualenv serves either.
The CTC-only engines (ONNX Runtime, GGUF) never reach this file: they need no
Python at all.

Protocol: a ``ready`` line on stdout once the model is in memory, then one
length-prefixed frame per request and one JSON object per line in reply. Frames
are packed ``(request_id, payload_size)`` followed by a canonical 16 kHz mono
PCM16 WAV.

Environment:
    GIGAAM_ENGINE       "onnx-asr" (default) or "mlx"
    GIGAAM_E2E_MODEL    "ctc" (faster) or "rnnt" (more accurate) for the MLX path
    GIGAAM_ORT_MODEL    onnx-asr model name; default gigaam-v3-e2e-ctc
    GIGAAM_ORT_PROVIDERS comma-separated onnxruntime providers; default CPU
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import time

# Big-endian so the Node side reads it without byte-order guesswork.
FRAME_HEADER = struct.Struct(">II")


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load_onnx_asr(model_name: str, providers: str):
    import onnx_asr

    selected = [p.strip() for p in providers.split(",") if p.strip()]
    model = onnx_asr.load_model(model_name, providers=selected)
    return lambda path: model.recognize(path)


def load_mlx(model_type: str):
    import gigaam_mlx

    model, tokenizer = gigaam_mlx.load_model(model_type)
    return lambda path: gigaam_mlx.transcribe(model, tokenizer, path)


def main() -> int:
    engine = os.environ.get("GIGAAM_ENGINE", "onnx-asr")
    started = time.perf_counter()
    try:
        if engine == "mlx":
            variant = os.environ.get("GIGAAM_E2E_MODEL", "ctc")
            recognize = load_mlx(variant)
            label = f"mlx:{variant}"
        else:
            model_name = os.environ.get("GIGAAM_ORT_MODEL", "gigaam-v3-e2e-ctc")
            providers = os.environ.get("GIGAAM_ORT_PROVIDERS", "CPUExecutionProvider")
            recognize = load_onnx_asr(model_name, providers)
            label = f"{engine}:{model_name}"
    except Exception as error:  # noqa: BLE001 - reported to the host verbatim
        sys.stderr.write(f"gigaam sidecar: cannot load {engine}: {error}\n")
        return 2

    # Announce readiness only after the weights are in memory, so the host's
    # warm-up ping means "ready" rather than "a process exists".
    emit(
        {
            "id": 0,
            "ok": True,
            "ready": True,
            "engine": engine,
            "model": label,
            "loadSeconds": time.perf_counter() - started,
        }
    )

    stdin = sys.stdin.buffer
    while True:
        header = stdin.read(FRAME_HEADER.size)
        if len(header) < FRAME_HEADER.size:
            break
        request_id, size = FRAME_HEADER.unpack(header)
        frame = stdin.read(size)
        if len(frame) < size:
            break
        temporary = None
        try:
            audio_seconds = (len(frame) - 44) / 32000 if len(frame) > 44 else 0.0
            # Both backends take a path and decode it with ffmpeg, which is
            # already inside the measured cost. Staying on their public API
            # avoids depending on internals that move between releases.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(frame)
                temporary = handle.name
            started = time.perf_counter()
            text = recognize(temporary)
            emit(
                {
                    "id": request_id,
                    "ok": True,
                    "text": (text or "").strip(),
                    "audioSeconds": audio_seconds,
                    "inferenceSeconds": time.perf_counter() - started,
                }
            )
        except Exception as error:  # noqa: BLE001 - one bad request must not kill the sidecar
            emit({"id": request_id, "ok": False, "error": f"gigaam {engine}: {error}"})
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
