/**
 * Worker process for the GigaAM provider.
 *
 * Owns the onnxruntime sessions so the host plugin keeps inference out of its
 * own event loop and can release ~430 MB of RSS by terminating the process.
 * Options arrive once through `GIGAAM_WORKER_OPTIONS`. Protocol: the parent
 * sends `{ id, type: "ping" }` or `{ id, type: "transcribe", wav }` and gets
 * `{ id, ok: true, ... }` or `{ id, ok: false, error }` back.
 */
import { InferenceSession, Tensor } from "onnxruntime-node";
import { assembleText, greedyCtc, loadVocab, parseWav } from "./decode.js";

/** Temporal subsampling of the GigaAM v3 encoder, from the exported model config. */
const SUBSAMPLING_FACTOR = 4;

const options = JSON.parse(process.env.GIGAAM_WORKER_OPTIONS ?? "{}");
let sessions = null;
let vocab = null;

async function ensureSessions() {
  if (sessions !== null) return sessions;
  vocab = loadVocab(options.vocabPath);
  const sessionOptions = {
    executionProviders: ["cpu"],
    interOpNumThreads: options.threads ?? 2,
    intraOpNumThreads: options.threads ?? 2,
  };
  const [feature, ctc] = await Promise.all([
    InferenceSession.create(options.featuresPath, sessionOptions),
    InferenceSession.create(options.modelPath, sessionOptions),
  ]);
  sessions = { feature, ctc };
  return sessions;
}

/**
 * Transcribe one canonical WAV recording.
 *
 * @param {Uint8Array} wav - canonical 16 kHz mono PCM16 WAV
 * @param {{ feature: import("onnxruntime-node").InferenceSession, ctc: import("onnxruntime-node").InferenceSession }} loaded
 * @returns {Promise<{ text: string, audioSeconds: number, inferenceSeconds: number }>}
 */
async function transcribe(wav, loaded) {
  const started = performance.now();
  const { pcm, seconds } = parseWav(wav);
  const { feature, ctc } = loaded;
  const features = await feature.run({
    [feature.inputNames[0]]: new Tensor("float32", pcm, [1, pcm.length]),
    [feature.inputNames[1]]: new Tensor("int64", BigInt64Array.from([BigInt(pcm.length)]), [1]),
  });
  const featureTensor = features[feature.outputNames[0]];
  const featureLengths = features[feature.outputNames[1]];
  const output = await ctc.run({
    [ctc.inputNames[0]]: featureTensor,
    [ctc.inputNames[1]]: featureLengths,
  });
  const logProbs = output[ctc.outputNames[0]];
  const vocabSize = logProbs.dims[logProbs.dims.length - 1];
  const frames = logProbs.dims[logProbs.dims.length - 2];
  const featuresLength = Number(featureLengths.data[0]);
  const expected = Math.floor((featuresLength - 1) / SUBSAMPLING_FACTOR) + 1;
  const usable = Math.max(0, Math.min(expected, frames));
  const text = assembleText(greedyCtc(logProbs.data, vocabSize, usable, vocab.blankId), vocab.tokens);
  return { text, audioSeconds: seconds, inferenceSeconds: (performance.now() - started) / 1000 };
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
        await ensureSessions();
        process.send({ id, ok: true, ready: true });
      } else if (type === "transcribe") {
        const loaded = await ensureSessions();
        process.send({ id, ok: true, ...(await transcribe(toBytes(message.wav), loaded)) });
      } else if (type === "shutdown") {
        process.exit(0);
      }
    } catch (error) {
      process.send({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
    }
  })();
});

process.on("uncaughtException", (error) => {
  process.send({ id: 0, ok: false, error: `gigaam worker crashed: ${error.message}` });
  process.exit(1);
});
