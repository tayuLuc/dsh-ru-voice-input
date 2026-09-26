/**
 * Metal worker for the GigaAM provider.
 *
 * Same wire protocol as the ONNX worker, different engine: a GigaAM v3 CTC GGUF
 * model through transcribe.cpp, whose encoder runs on the GPU. Measured on this
 * Apple M1 host (27.09) on the same 28.6 s Russian recording:
 *
 *   onnxruntime int8, CPU  RTF 0.041
 *   GGUF Q8_0, CPU         RTF 0.048
 *   GGUF Q8_0, Metal       RTF 0.016   <- this worker
 *
 * The model is loaded lazily on the first `ping` and released with the process,
 * so the host keeps the same "terminate the worker to free GPU memory" contract.
 */
import { TranscribeModel } from "transcribe-cpp";
import { parseWav } from "./decode.js";

/** transcribe.cpp consumes planar float32 samples, not the WAV container. */
const SAMPLE_RATE = 16000;

const options = JSON.parse(process.env.GIGAAM_WORKER_OPTIONS ?? "{}");
let model = null;
let device = null;

async function ensureModel() {
  if (model !== null) return model;
  const loaded = await TranscribeModel.load(options.modelPath, {
    backend: options.backend ?? "metal",
    n_threads: options.threads ?? 2
  });
  model = loaded;
  device = loaded.device;
  return model;
}

/**
 * Transcribe one canonical WAV recording.
 *
 * @param {Uint8Array} wav - canonical 16 kHz mono PCM16 WAV
 * @returns {Promise<{ text: string, audioSeconds: number, inferenceSeconds: number }>}
 */
async function transcribe(wav) {
  const started = performance.now();
  // parseWav already yields planar float32 in [-1, 1]; rescaling here would
  // shrink the signal by 32768 and the recogniser would return silence.
  const { pcm, seconds } = parseWav(wav);
  const loaded = await ensureModel();
  const result = await loaded.transcribe(pcm, { language: options.language ?? "ru", sample_rate: SAMPLE_RATE });
  return {
    text: (result.text ?? "").trim(),
    audioSeconds: seconds,
    inferenceSeconds: (performance.now() - started) / 1000
  };
}

/**
 * Normalise an incoming recording.
 *
 * `child_process.fork` defaults to JSON serialisation, where a Buffer arrives as
 * `{ type: "Buffer", data: [...] }`; with the advanced serialisation it stays a
 * typed array view. Accept both instead of trusting one transport.
 *
 * @param {unknown} value
 * @returns {Uint8Array}
 */
function toBytes(value) {
  if (value instanceof Uint8Array) return value;
  if (Array.isArray(value?.data)) return Uint8Array.from(value.data);
  if (Array.isArray(value)) return Uint8Array.from(value);
  throw new Error("gigaam: recording payload is not a byte sequence");
}

process.on("message", (message) => {
  void (async () => {
    const { id, type } = message ?? {};
    try {
      if (type === "ping") {
        await ensureModel();
        process.send({ id, ok: true, ready: true, device: device?.name ?? null, deviceType: device?.deviceType ?? null });
      } else if (type === "transcribe") {
        const loaded = await ensureModel();
        process.send({ id, ok: true, ...(await transcribe(toBytes(message.wav), loaded)) });
      } else if (type === "shutdown") {
        model?.dispose?.();
        process.exit(0);
      }
    } catch (error) {
      process.send({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
    }
  })();
});

process.on("uncaughtException", (error) => {
  process.send({ id: 0, ok: false, error: `gigaam metal worker crashed: ${error.message}` });
  process.exit(1);
});
