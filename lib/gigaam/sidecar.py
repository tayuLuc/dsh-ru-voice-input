"""GigaAM v3 e2e sidecar for the DSH speech provider.

MLX port of GigaAM v3 (Salute Developers, MIT), packaged as `gigaam-mlx`. It is
the only engine in this family that emits punctuation and capitalisation: the
CTC heads the provider also supports have a 34-token letter vocabulary and
cannot produce either.

Protocol: on stdout one JSON object per line. On stdin a stream of frames,
each a packed ``(request_id, payload_size)`` header followed by the payload —
a canonical 16 kHz mono PCM16 WAV, the same bytes the Node provider receives
from the browser. The model loads once at startup and stays warm for the life
of the process; a ready line with ``id`` 0 is emitted once it is in memory.

Environment:
    GIGAAM_E2E_MODEL     "ctc" (default, faster) or "rnnt" (slower, more accurate)
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


def main() -> int:
    # The model package prints download progress to stderr; keep stdout clean so
    # the Node side can parse every line as JSON.
    model_type = os.environ.get("GIGAAM_E2E_MODEL", "ctc")
    try:
        import gigaam_mlx
    except Exception as error:  # noqa: BLE001 - reported to the host verbatim
        sys.stderr.write(f"gigaam e2e: gigaam-mlx is not importable: {error}\n")
        return 2

    stdout = sys.stdout
    stdin = sys.stdin.buffer

    def emit(payload: dict) -> None:
        stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stdout.flush()

    # Load before announcing readiness, so the host's warm-up ping really means
    # "the model is in memory" rather than "a process exists".
    started = time.perf_counter()
    model, tokenizer = gigaam_mlx.load_model(model_type)
    emit(
        {
            "id": 0,
            "ok": True,
            "ready": True,
            "model": model_type,
            "loadSeconds": time.perf_counter() - started,
        }
    )

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
            # The public entry point takes a path and decodes it with ffmpeg.
            # ffmpeg is already inside the measured cost, and it keeps this
            # sidecar on the package's supported API rather than its internals.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(frame)
                temporary = handle.name
            started = time.perf_counter()
            text = gigaam_mlx.transcribe(model, tokenizer, temporary)
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
            emit({"id": request_id, "ok": False, "error": f"gigaam e2e: {error}"})
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
