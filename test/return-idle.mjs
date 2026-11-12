/**
 * Behavioural checks for what a recording pays when it comes back after a pause.
 *
 * The complaint these cover is the composer row "Waking…" (ru: «Пробуждение…»)
 * appearing before the waveform: the engine had been released, so the next
 * recording pays the load. Three properties decide it, and all three are
 * provider-side — the UI only renders the phase it is told about:
 *
 *   1. `idleTimeoutMs: 0` is the default, so the engine is never released while
 *      the provider lives: there is no pause to wake up from.
 *   2. With a real idle window, a transcribe() after the release publishes
 *      `waking` and lands back at `ready` — that is the row, and its cost is the
 *      worker's load time.
 *   3. Unmounting the provider releases the worker even when `idleTimeoutMs` is
 *      0, because the default keeps it alive forever and the provider is gone.
 *      This is the leak the never-release mode used to have.
 *
 * The engine is picked from what this machine has, same rule as warmth.mjs: a
 * mock cannot show a load having finished.
 *
 * Run with: node test/return-idle.mjs [modelDirectory]
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

let failures = 0;
const check = async (name, run) => {
  try {
    await run();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures += 1;
    console.log(`FAIL ${name}: ${error.message}`);
  }
};

const modelDirectory = process.argv[2] ?? join(homedir(), ".dsh", "models", "gigaam-v3-ctc");
const ggufFile = join(modelDirectory, "gigaam-v3-ctc-Q8_0.gguf");
const onnxFile = join(modelDirectory, "v3_ctc.int8.onnx");
const python = join(homedir(), ".dsh", "venvs", "gigaam-mlx", "bin", "python");

function pickEngine() {
  if (existsSync(python)) return { engine: "mlx", why: `интерпретатор ${python}` };
  if (existsSync(onnxFile)) return { engine: "onnx", why: onnxFile };
  if (existsSync(ggufFile)) return { engine: "gguf", why: ggufFile };
  return undefined;
}

const chosen = pickEngine();
if (chosen === undefined) {
  console.log("ни один движок не готов на этой машине: нет ни интерпретатора, ни весов.");
  console.log(`Ожидалось одно из: ${python} / ${onnxFile} / ${ggufFile}`);
  process.exit(2);
}
console.log(`движок теста: ${chosen.engine} (${chosen.why})\n`);

/**
 * Canonical 16 kHz mono PCM16 WAV of near-silence.
 *
 * The negative control of provider.mjs reused here: the assertions are about
 * phases and processes, so the audio only has to be accepted by the decoder.
 * @param {number} seconds
 * @returns {Uint8Array}
 */
function silenceWav(seconds) {
  const samples = Math.floor(16000 * seconds);
  const bytes = new Uint8Array(44 + samples * 2);
  const view = new DataView(bytes.buffer);
  const ascii = (offset, text) => [...text].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
  ascii(0, "RIFF");
  view.setUint32(4, bytes.length - 8, true);
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
  view.setUint32(40, bytes.length - 44, true);
  return bytes;
}

const { apply, Config } = await import(new URL("../lib/gigaam/index.js", import.meta.url));

/**
 * Read a schema node's declared default through the schemastery meta chain.
 *
 * Read from the shipped schema, not restated here, so a default that moves in
 * the source moves this assertion with it.
 * @param {string} field
 * @returns {unknown}
 */
function schemaDefault(field) {
  let node = Config.dict[field];
  for (let hop = 0; hop < 8 && node !== undefined; hop += 1) {
    const meta = typeof node === "function" ? node.meta : node?.meta;
    if (meta !== undefined && "default" in meta) return meta.default;
    node = typeof node === "function" ? node.default : node?.default;
  }
  return undefined;
}

function mount(overrides = {}) {
  const registrations = [];
  const ctx = {
    speechToText: {
      register(provider) {
        registrations.push(provider);
        return () => registrations.splice(registrations.indexOf(provider), 1);
      },
      async configure() {},
      snapshot: () => ({ selection: { providerId: "warmth test", language: "ru" } }),
    },
    logger: { warn: (message) => console.log(`  warn: ${message}`) },
  };
  const config = {
    providerId: "gigaam-v3-ctc-local",
    displayName: "warmth test",
    modelDirectory,
    modelOrigin: "https://huggingface.co",
    modelRepo: "istupakov/gigaam-v3-onnx",
    modelRevision: "322c3b29492673eb7d0b434bfa9dfb8653e34d02",
    modelFile: "v3_ctc.int8.onnx",
    vocabFile: "v3_vocab.txt",
    modelSha256: "ceb61454e2e1a2dec5872cbac1de0fe0a4271d1148f6b26b5bda53ff30a12acd",
    vocabSha256: "a9143c30844d3c0bee3e9e927e4084774eb1b9eeaafc473b2c4521e4911a7c07",
    ggufOrigin: "https://huggingface.co",
    ggufRepo: "handy-computer/gigaam-v3-ctc-gguf",
    ggufRevision: "696b1bc14be5a4c423090bdc31da27793def4065",
    ggufFile: "gigaam-v3-ctc-Q8_0.gguf",
    ggufSha256: "71e5c82890e9e243a6bd7575f129f5d1bd2c3ca3ae79aab75cfb1a6934c6a62b",
    threads: 2,
    idleTimeoutMs: 0,
    language: "ru",
    engine: chosen.engine,
    e2ePythonPath: "",
    e2eModel: "ctc",
    ortModel: "gigaam-v3-e2e-ctc",
    ortProviders: "CPUExecutionProvider",
    backend: "cpu",
    lexiconPath: "/nonexistent-stt-lexicon.json",
    lexiconThreshold: 0.8,
    setAsDefaultProvider: false,
    preload: false,
    ...overrides,
  };
  const dispose = apply(ctx, config);
  return { provider: registrations[0], dispose };
}

/** @param {object} provider @param {string} wanted @param {number} attempts */
const waitFor = async (provider, wanted, attempts = 400) => {
  let phase = provider.preparation.snapshot().phase;
  for (let attempt = 0; attempt < attempts && phase !== wanted; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 50));
    phase = provider.preparation.snapshot().phase;
  }
  return phase;
};

/** Every live pid whose command line mentions one of this provider's workers. */
function workerPids() {
  try {
    const out = execFileSync("ps", ["-axo", "pid=,command="], { encoding: "utf8" });
    return out
      .split("\n")
      .filter((line) => /gigaam\/(worker|metal-worker|e2e-worker)\.js/.test(line))
      .map((line) => Number(line.trim().split(/\s+/)[0]))
      .filter((pid) => Number.isFinite(pid) && pid !== process.pid);
  } catch {
    return [];
  }
}

await check("по умолчанию движок не выгружается: паузы, из которой надо просыпаться, нет", () => {
  assert.equal(schemaDefault("idleTimeoutMs"), 0, "idleTimeoutMs должен иметь default 0");
});

await check("после прогрева движок стоит в ready, а не в standby", async () => {
  const { provider, dispose } = mount({ preload: true, idleTimeoutMs: 0 });
  const phase = await waitFor(provider, "ready");
  await new Promise((resolve) => setTimeout(resolve, 400));
  assert.equal(provider.preparation.snapshot().phase, "ready", "после прогрева движок разогрет");
  assert.equal(phase, "ready");
  await provider.preparation.cancel();
  await dispose();
});

await check("запись после простоя публикует waking и возвращается в ready", async () => {
  const { provider, dispose } = mount({ preload: true, idleTimeoutMs: 250 });
  assert.equal(await waitFor(provider, "ready"), "ready", "прогрев не дошёл до ready");
  await new Promise((resolve) => setTimeout(resolve, 450));
  assert.equal(provider.preparation.snapshot().phase, "standby", "движок должен был выгрузиться по таймеру");
  const phases = [];
  const unsubscribe = provider.preparation.subscribe(() => phases.push(provider.preparation.snapshot().phase));
  const started = Date.now();
  await provider.transcribe({ audio: silenceWav(0.5), language: "ru" }, new AbortController().signal);
  unsubscribe();
  const wakeSeconds = (Date.now() - started) / 1000;
  assert.ok(phases.includes("waking"), `должен был быть опубликован waking, опубликовано: ${phases.join(" → ")}`);
  assert.equal(provider.preparation.snapshot().phase, "ready", "после записи движок снова готов");
  console.log(`  wake+decode: ${wakeSeconds.toFixed(2)} с на движке ${chosen.engine}`);
});

await check("снятие регистрации глушит воркер, даже когда idleTimeoutMs = 0", async () => {
  const { provider, dispose } = mount({ preload: true, idleTimeoutMs: 0 });
  assert.equal(await waitFor(provider, "ready"), "ready", "прогрев не дошёл до ready");
  const live = workerPids();
  assert.ok(live.length > 0, "прогретый движок должен был оставить дочерний процесс");
  await dispose();
  for (let attempt = 0; attempt < 60 && workerPids().some((pid) => live.includes(pid)); attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  const survivors = workerPids().filter((pid) => live.includes(pid));
  assert.equal(survivors.length, 0, `после снятия регистрации живы воркеры: ${survivors.join(", ")}`);
});

if (failures > 0) {
  console.log(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("all checks passed");
