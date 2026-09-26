/**
 * Live check of the provider against a real recording.
 *
 * Usage: node scripts/selftest.mjs <recording.wav> [modelDirectory]
 *
 * Boots the same child worker the plugin uses, transcribes the file, and prints
 * the transcript with measured audio time, inference time, and worker RSS. A
 * non-zero exit code means the check failed; an empty transcript is reported as
 * a failure because silence and "model cannot do this language" look identical
 * otherwise.
 */
import { fork } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join, resolve } from "node:path";

const wavPath = process.argv[2];
if (wavPath === undefined) {
  console.error("usage: node scripts/selftest.mjs <recording.wav> [modelDirectory] [engine: onnx|metal]");
  process.exit(2);
}
const engine = process.argv[4] ?? "onnx";
const directory = resolve(process.argv[3] ?? join(process.env.HOME ?? "/root", ".dsh", "models", "gigaam-v3-ctc"));
const modelName = engine === "metal" ? "gigaam-v3-ctc-Q8_0.gguf" : "v3_ctc.int8.onnx";
const modelPath = resolve(directory, modelName);
if (!existsSync(modelPath)) {
  console.error(`selftest: model not found at ${modelPath}`);
  console.error("prepare it first: the plugin downloads the pinned file into <modelDirectory>");
  process.exit(2);
}

const workerFile = engine === "metal" ? "metal-worker.js" : "worker.js";
const worker = fork(fileURLToPath(new URL(`../lib/gigaam/${workerFile}`, import.meta.url)), [], {
  stdio: ["ignore", "inherit", "inherit", "ipc"],
  serialization: "advanced",
  env: {
    ...process.env,
    GIGAAM_WORKER_OPTIONS: JSON.stringify({
      modelPath,
      vocabPath: fileURLToPath(new URL("../lib/assets/v3_vocab.txt", import.meta.url)),
      featuresPath: fileURLToPath(new URL("../lib/assets/gigaam_v3_features.onnx", import.meta.url)),
      threads: 2,
      backend: "metal",
      language: "ru",
    }),
  },
});

const wav = (await import("node:fs")).readFileSync(wavPath);
/**
 * Ping before transcribing, exactly as the provider does. Both engines load
 * their model on the first ping, so a direct transcribe would fold the one-off
 * load into the measured inference time and make Metal look as slow as CPU.
 */
let ready = false;
const runTranscribe = () => {
  worker.send({ id: 2, type: "transcribe", wav }, (error) => {
    if (error) {
      console.error(`selftest: cannot send the recording: ${error.message}`);
      worker.kill();
      process.exit(1);
    }
  });
};
const started = Date.now();
worker.send({ id: 1, type: "ping" }, (error) => {
  if (error) {
    console.error(`selftest: cannot ping the worker: ${error.message}`);
    worker.kill();
    process.exit(1);
  }
});

worker.on("message", (message) => {
  if (message.ok !== true) {
    console.error(`selftest: ${message.error}`);
    worker.kill();
    process.exit(1);
  }
  if (message.id === 1) {
    ready = true;
    console.log(`load=${((Date.now() - started) / 1000).toFixed(2)}s device=${message.deviceType ?? "cpu"}/${message.device ?? "cpu"}`);
    runTranscribe();
    return;
  }
  if (!ready) return;
  console.log(`audio=${message.audioSeconds.toFixed(2)}s inference=${message.inferenceSeconds.toFixed(2)}s`);
  console.log(`rtf=${(message.inferenceSeconds / message.audioSeconds).toFixed(3)}`);
  console.log(`transcript: ${message.text}`);
  if (message.text.trim() === "") {
    console.error("selftest: empty transcript — the model produced no tokens for this recording");
    worker.kill();
    process.exit(1);
  }
  worker.kill();
  process.exit(0);
});

worker.on("exit", (code) => {
  if (code !== 0 && code !== null) {
    console.error(`selftest: worker exited with code ${code}`);
    process.exit(1);
  }
});
