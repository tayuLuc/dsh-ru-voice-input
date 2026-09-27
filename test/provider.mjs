/**
 * Integration check for the shipped GigaAM provider.
 *
 * Loads `lib/gigaam/index.js` unmodified and drives it the way the host does:
 * register into a stand-in `ctx.speechToText`, prepare, transcribe. This is the
 * only check that covers the parts a worker-level test cannot see — the config
 * contract, the preparation lifecycle, the model lookup, and the provider info
 * the settings UI reads.
 *
 * It needs the model already present in a directory; nothing is downloaded here.
 * Run with: node test/provider.mjs <recording.wav> [modelDirectory]
 */
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const wavPath = process.argv[2];
const modelDirectory = resolve(process.argv[3] ?? join(process.env.HOME ?? "/root", ".dsh", "models", "gigaam-v3-ctc"));
const engine = process.argv[4] ?? "metal";
const ggufFile = "gigaam-v3-ctc-Q8_0.gguf";
const onnxFile = "v3_ctc.int8.onnx";
const modelName = engine === "metal" ? ggufFile : onnxFile;

if (wavPath === undefined) {
  console.error("usage: node test/provider.mjs <recording.wav> [modelDirectory] [engine: metal|onnx]");
  process.exit(2);
}
if (!existsSync(join(modelDirectory, modelName))) {
  console.error(`provider: model ${modelName} is not in ${modelDirectory}`);
  console.error("prepare it first: the plugin downloads the pinned file into <modelDirectory>");
  process.exit(2);
}

let failures = 0;
const check = (name, run) => {
  try {
    run();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures += 1;
    console.log(`FAIL ${name}: ${error.message}`);
  }
};

const { apply } = await import(new URL("../lib/gigaam/index.js", import.meta.url));

/** Minimal host surface: the provider only touches these. */
const registrations = [];
const settingsWrites = [];
const ctx = {
  speechToText: {
    register(provider) {
      registrations.push(provider);
      return () => {
        registrations.splice(registrations.indexOf(provider), 1);
      };
    },
    async configure(patch) {
      settingsWrites.push(patch);
    }
  },
  logger: { warn: (message) => console.log(`  warn: ${message}`) }
};

const config = {
  providerId: "gigaam-v3-ctc-local",
  displayName: "GigaAM v3 CTC int8 (локально)",
  modelDirectory,
  modelOrigin: "https://huggingface.co",
  modelRepo: "istupakov/gigaam-v3-onnx",
  modelRevision: "322c3b29492673eb7d0b434bfa9dfb8653e34d02",
  modelFile: onnxFile,
  vocabFile: "v3_vocab.txt",
  modelSha256: "ceb61454e2e1a2dec5872cbac1de0fe0a4271d1148f6b26b5bda53ff30a12acd",
  vocabSha256: "a9143c30844d3c0bee3e9e927e4084774eb1b9eeaafc473b2c4521e4911a7c07",
  threads: 2,
  // A short idle window so the worker self-releases and the test process can
  // exit: with 0 the provider deliberately keeps the worker loaded forever, and
  // the live IPC channel to that child pins the parent event loop.
  idleTimeoutMs: 250,
  language: "ru",
  engine,
  backend: "metal",
  ggufOrigin: "https://huggingface.co",
  ggufRepo: "handy-computer/gigaam-v3-ctc-gguf",
  ggufRevision: "696b1bc14be5a4c423090bdc31da27793def4065",
  ggufFile,
  ggufSha256: "71e5c82890e9e243a6bd7575f129f5d1bd2c3ca3ae79aab75cfb1a6934c6a62b",
  setAsDefaultProvider: true
};

const dispose = apply(ctx, config);
const provider = registrations[0];

check("registers exactly one provider", () => {
  assert.equal(registrations.length, 1);
});

check("info satisfies the registry contract", () => {
  assert.equal(provider.info.id, "gigaam-v3-ctc-local");
  assert.equal(provider.info.location, "host-local");
  // "auto" must be present, or a profile left on the default language cannot
  // select this provider at all.
  assert.ok(provider.info.languages.includes("ru"), "ru advertised");
  assert.ok(provider.info.languages.includes("auto"), "auto advertised");
  assert.ok(Array.isArray(provider.info.downloadSources) && provider.info.downloadSources.length > 0, "download sources listed");
  assert.ok(provider.info.setupEstimate.recommendedDiskBytes > 0, "disk estimate present");
  assert.ok(provider.info.setupEstimate.expectedMemoryBytes > 0, "memory estimate present");
});

check("asks the registry to select itself", () => {
  assert.equal(settingsWrites.at(-1)?.providerId, "gigaam-v3-ctc-local");
  assert.equal(settingsWrites.at(-1)?.language, "ru");
});

check("starts in standby when the model is already on disk", () => {
  assert.equal(provider.preparation.snapshot().phase, "standby");
});

await provider.preparation.prepare();
check("preparation reaches ready", () => {
  assert.equal(provider.preparation.snapshot().phase, "ready");
});

const controller = new AbortController();
const result = await provider.transcribe({ audio: readFileSync(wavPath), language: "ru" }, controller.signal);

check("transcribe returns a non-empty Russian transcript", () => {
  assert.ok(typeof result.text === "string" && result.text.length > 0, "transcript is not empty");
  assert.ok(result.audioSeconds > 0, "audio duration reported");
  assert.ok(result.inferenceSeconds > 0, "inference duration reported");
});

const silence = new Uint8Array(44 + 32000 * 2);
const view = new DataView(silence.buffer);
const ascii = (offset, text) => [...text].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
ascii(0, "RIFF");
view.setUint32(4, silence.length - 8, true);
ascii(8, "WAVE");
ascii(12, "fmt ");
view.setUint32(16, 16, true);
view.setUint16(20, 1, true);
view.setUint16(22, 1, true);
view.setUint32(24, 16000, true);
view.setUint32(28, 32000, true);
view.setUint16(32, 2, true);
view.setUint16(34, 16, true);
ascii(36, "data");
view.setUint32(40, silence.length - 44, true);

let silenceRejected = false;
try {
  const quiet = await provider.transcribe({ audio: silence, language: "ru" }, controller.signal);
  silenceRejected = (quiet.text ?? "").trim() === "";
} catch (error) {
  silenceRejected = true;
}
check("silence yields no tokens, so a broken model cannot pass as working", () => {
  assert.ok(silenceRejected, "silence produced text");
});

const rtf = result.inferenceSeconds / result.audioSeconds;
console.log(`engine=${engine} audio=${result.audioSeconds.toFixed(2)}s inference=${result.inferenceSeconds.toFixed(2)}s rtf=${rtf.toFixed(3)}`);
console.log(`transcript: ${result.text.slice(0, 90)}`);

await dispose();
if (failures > 0) {
  console.log(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("all checks passed");
