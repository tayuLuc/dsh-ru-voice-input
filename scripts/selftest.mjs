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
import { resolve } from "node:path";

const wavPath = process.argv[2];
if (wavPath === undefined) {
  console.error("usage: node scripts/selftest.mjs <recording.wav> [modelDirectory]");
  process.exit(2);
}
const directory = resolve(process.argv[3] ?? "/tmp/hf/hub/models--istupakov--gigaam-v3-onnx/snapshots/current");
const modelPath = resolve(directory, "v3_ctc.int8.onnx");
if (!existsSync(modelPath)) {
  console.error(`selftest: model not found at ${modelPath}`);
  console.error("prepare it first: the plugin downloads the pinned file into <modelDirectory>");
  process.exit(2);
}

const worker = fork(fileURLToPath(new URL("../lib/gigaam/worker.js", import.meta.url)), [], {
  stdio: ["ignore", "inherit", "inherit", "ipc"],
  serialization: "advanced",
  env: {
    ...process.env,
    GIGAAM_WORKER_OPTIONS: JSON.stringify({
      modelPath,
      vocabPath: fileURLToPath(new URL("../lib/assets/v3_vocab.txt", import.meta.url)),
      featuresPath: fileURLToPath(new URL("../lib/assets/gigaam_v3_features.onnx", import.meta.url)),
      threads: 2,
    }),
  },
});

const wav = (await import("node:fs")).readFileSync(wavPath);
worker.send({ id: 1, type: "transcribe", wav }, (error) => {
  if (error) {
    console.error(`selftest: cannot send the recording: ${error.message}`);
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
