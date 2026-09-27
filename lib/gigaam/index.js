/**
 * dsh-stt-gigaam — local Russian speech-to-text provider for the DeepSeek Harness.
 *
 * Registers one recognizer into `ctx.speechToText`: GigaAM v3 CTC (int8) running
 * on the host CPU through onnxruntime-node, inside a child process the plugin
 * owns and can terminate. It exists because the official Voice input provider
 * ships a SenseVoice engine whose released checkpoint has no Russian: on a
 * Russian recording it returns an empty transcript while still holding half a
 * gigabyte of RAM.
 *
 * Everything tunable is a Config field; the model artefacts are revision-pinned
 * and digest-verified before use, and preparation never touches a
 * half-downloaded file.
 */
import { fork } from "node:child_process";
import { existsSync } from "node:fs";
import { join, resolve as resolvePath } from "node:path";
import { fileURLToPath } from "node:url";
import z from "@deepseek-ai/schemastery";
import { ModelSource } from "./model-source.js";

export const name = "stt-gigaam";
/** Wait for the speech registry; unload cleanly when the service disappears. */
export const inject = ["speechToText"];

const FEATURES_PATH = fileURLToPath(new URL("../assets/gigaam_v3_features.onnx", import.meta.url));
const VOCAB_PATH = fileURLToPath(new URL("../assets/v3_vocab.txt", import.meta.url));
const WORKER_PATH = fileURLToPath(new URL("./worker.js", import.meta.url));
/** The GGUF/Metal engine; same wire protocol, different runtime. */
const METAL_WORKER_PATH = fileURLToPath(new URL("./metal-worker.js", import.meta.url));

/** Measured on an Apple M1 host: ~225 MB on disk, ~430 MB RSS, RTF 0.04. */
const SETUP_ESTIMATE = {
  recommendedDiskBytes: 240_000_000,
  expectedMemoryBytes: 500_000_000,
  minimumMinutes: 1,
  maximumMinutes: 5,
};

/** Measured on the same host for the GGUF Q8_0 file: 272 MB on disk, RTF 0.016. */
const METAL_SETUP_ESTIMATE = {
  recommendedDiskBytes: 290_000_000,
  expectedMemoryBytes: 500_000_000,
  minimumMinutes: 1,
  maximumMinutes: 5,
};

export const Config = z.object({
  /** Provider identity the registry shows and `defaultProvider` can select. */
  providerId: z.string().default("gigaam-v3-ctc-local"),
  /** Human-readable provider name in the plugin settings. */
  displayName: z.string().default("GigaAM v3 CTC int8 (локально)"),
  /** Directory for the managed model files; defaults to <dsh home>/models/gigaam-v3-ctc. */
  modelDirectory: z.string(),
  /** Download origin for the pinned model; any Hugging Face compatible host. */
  modelOrigin: z.string().default("https://huggingface.co"),
  /** Repository holding the exported ONNX weights. */
  modelRepo: z.string().default("istupakov/gigaam-v3-onnx"),
  /** Pinned revision; changing it invalidates the pinned digests below. */
  modelRevision: z.string().default("322c3b29492673eb7d0b434bfa9dfb8653e34d02"),
  /** Model file name inside the repository. */
  modelFile: z.string().default("v3_ctc.int8.onnx"),
  /** Vocabulary file name inside the repository (verified, then unused: the bundled copy is loaded). */
  vocabFile: z.string().default("v3_vocab.txt"),
  /** Expected SHA-256 of the model file at the pinned revision. */
  modelSha256: z.string().default("ceb61454e2e1a2dec5872cbac1de0fe0a4271d1148f6b26b5bda53ff30a12acd"),
  /** Expected SHA-256 of the repository vocabulary, used as the download canary. */
  vocabSha256: z.string().default("a9143c30844d3c0bee3e9e927e4084774eb1b9eeaafc473b2c4521e4911a7c07"),
  /** CPU threads for onnxruntime; 2 keeps a dictation turn off the critical path. */
  threads: z.natural().default(2),
  /** Release the worker (and its RAM) after this idle period; 0 keeps it loaded. */
  idleTimeoutMs: z.natural().default(120_000),
  /** Language hint this provider accepts. GigaAM v3 CTC is a Russian character model. */
  language: z.string().default("ru"),
  /**
   * Which engine runs the recognizer.
   *
   * "metal" loads a GigaAM v3 CTC GGUF model through transcribe.cpp and puts the
   * encoder on the GPU; measured RTF 0.016 against 0.041 for the ONNX CPU path on
   * this host. "onnx" keeps the int8 onnxruntime path, which is the fallback when
   * no Metal device is available.
   */
  engine: z.union(["metal", "onnx"]).default("metal"),
  /** Compute backend handed to transcribe.cpp; "auto" lets it choose. */
  backend: z.string().default("metal"),
  /** Download origin for the pinned GGUF weights. */
  ggufOrigin: z.string().default("https://huggingface.co"),
  /** Repository holding the exported GGUF weights. */
  ggufRepo: z.string().default("handy-computer/gigaam-v3-ctc-gguf"),
  /** Pinned revision of the GGUF repository; changing it invalidates its digest. */
  ggufRevision: z.string().default("696b1bc14be5a4c423090bdc31da27793def4065"),
  /** GGUF file name inside the repository. */
  ggufFile: z.string().default("gigaam-v3-ctc-Q8_0.gguf"),
  /** Expected SHA-256 of the GGUF file at the pinned revision. */
  ggufSha256: z.string().default("71e5c82890e9e243a6bd7575f129f5d1bd2c3ca3ae79aab75cfb1a6934c6a62b"),
  /** Select this provider as the registry default once it registers. */
  setAsDefaultProvider: z.boolean().default(true),
});

/** Resolve the default model directory without hardcoding a home path. */
function defaultModelDirectory() {
  const home = process.env.DSH_HOME;
  if (home !== undefined && home !== "") return join(home, "models", "gigaam-v3-ctc");
  return join(process.env.HOME ?? ".", ".dsh", "models", "gigaam-v3-ctc");
}

/**
 * Build one speech provider bound to the plugin config.
 *
 * @param {import("@deepseek-ai/cordis").Context} ctx
 * @param {z.infer<typeof Config>} config
 * @returns {import("@deepseek-ai/dsh-experimental-speech-to-text").SpeechProvider}
 */
function createProvider(ctx, config) {
  const directory = resolvePath(config.modelDirectory || defaultModelDirectory());
  const source = new ModelSource({
    directory,
    origin: config.modelOrigin,
    repo: config.modelRepo,
    revision: config.modelRevision,
  });

  /** @type {{ phase: string, resource?: string, completedBytes?: number, totalBytes?: number, startedAt?: number, message?: string, steps?: { kind: string, status: string }[] }} */
  let state = { phase: "unprepared", steps: [] };
  const listeners = new Set();
  let preparation = null;
  let worker = null;
  let workerReady = null;
  let requestId = 0;
  let idleTimer = null;

  const publish = (next) => {
    state = next;
    for (const listener of listeners) listener();
  };

  const modelPath = source.path(config.modelFile);
  const vocabPath = source.path(config.vocabFile);

  /** The active engine decides which worker, which model and which estimate apply. */
  const useMetal = config.engine === "metal";
  const ggufSource = new ModelSource({
    directory,
    origin: config.ggufOrigin,
    repo: config.ggufRepo,
    revision: config.ggufRevision,
  });
  const ggufPath = ggufSource.path(config.ggufFile);
  const activeModelPath = useMetal ? ggufPath : modelPath;
  const activeFileName = useMetal ? config.ggufFile : config.modelFile;

  /**
   * Which managed artefacts are already usable.
   *
   * Synchronous on purpose: the activation state must be truthful on the very
   * first snapshot. An `async` wrapper here made every reader that sampled the
   * state before the microtask ran see `unprepared`, which the settings UI takes
   * to mean "no model present" and answers with a download prompt even when the
   * weights sit on disk.
   */
  function inspect() {
    // The bundled feature graph and vocabulary belong to the ONNX decode path;
    // the GGUF engine carries its own tokenizer and does not read them.
    if (!useMetal) {
      if (!existsSync(FEATURES_PATH)) throw new Error("gigaam: bundled feature graph is missing from the package");
      if (!existsSync(VOCAB_PATH)) throw new Error("gigaam: bundled vocabulary is missing from the package");
    }
    if (!existsSync(activeModelPath)) return { ready: false, missing: activeFileName };
    return { ready: true, missing: undefined };
  }

  /** Make the model file for the active engine available, digest-verified. */
  async function ensureModel(signal) {
    if (useMetal) {
      return await ggufSource.ensure(config.ggufFile, config.ggufSha256, signal);
    }
    const model = await source.ensure(config.modelFile, config.modelSha256, signal);
    // The repository vocabulary is fetched as the download canary, then
    // discarded: the bundled copy is what the worker loads.
    const vocab = await source.ensure(config.vocabFile, config.vocabSha256, signal);
    if (model.bytes === 0 || vocab.bytes === 0) throw new Error("gigaam: downloaded file is empty");
    return model;
  }

  async function startWorker() {
    if (worker !== null) return worker;
    const child = fork(useMetal ? METAL_WORKER_PATH : WORKER_PATH, [], {
      stdio: ["ignore", "ignore", "ignore", "ipc"],
      serialization: "advanced",
      env: {
        ...process.env,
        GIGAAM_WORKER_OPTIONS: JSON.stringify({
          modelPath: activeModelPath,
          vocabPath: VOCAB_PATH,
          featuresPath: FEATURES_PATH,
          threads: config.threads,
          backend: config.backend,
          language: config.language,
        }),
      },
    });
    const ready = new Promise((resolveReady, rejectReady) => {
      const id = ++requestId;
      const onMessage = (message) => {
        if (message?.id !== id) return;
        child.off("message", onMessage);
        if (message.ok) resolveReady(child);
        else rejectReady(new Error(String(message.error)));
      };
      child.on("message", onMessage);
      child.once("error", rejectReady);
      child.once("exit", (code) => rejectReady(new Error(`gigaam worker exited before loading (code ${code})`)));
      child.send({ id, type: "ping" });
    });
    worker = child;
    workerReady = ready;
    child.once("exit", () => {
      if (worker === child) {
        worker = null;
        workerReady = null;
        if (state.phase === "ready" || state.phase === "waking") publish({ phase: "standby", steps: state.steps });
      }
    });
    return ready;
  }

  function stopWorker() {
    if (idleTimer !== null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
    const child = worker;
    worker = null;
    workerReady = null;
    if (child !== null) child.kill("SIGKILL");
  }

  function scheduleIdleRelease() {
    if (config.idleTimeoutMs === 0) return;
    if (idleTimer !== null) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      idleTimer = null;
      if (state.phase === "ready") {
        stopWorker();
        publish({ phase: "standby", steps: state.steps });
      }
    }, config.idleTimeoutMs);
    idleTimer.unref?.();
  }

  const provider = {
    info: {
      id: config.providerId,
      name: config.displayName,
      location: "host-local",
      // GigaAM v3 is a Russian character model, so "auto" is accepted as an
      // alias for Russian. Declaring it matters: the registry validates the
      // CURRENT language against the NEWLY selected provider, so a profile still
      // on the default "auto" would otherwise be unable to switch to this
      // provider at all.
      languages: [...new Set([config.language, "auto"])],
      downloadSources: [...new Set([new URL(useMetal ? config.ggufOrigin : config.modelOrigin).origin])],
      setupEstimate: useMetal ? METAL_SETUP_ESTIMATE : SETUP_ESTIMATE,
    },
    preparation: {
      snapshot: () => state,
      subscribe(listener) {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      prepare() {
        if (preparation !== null) return preparation;
        const controller = new AbortController();
        preparation = (async () => {
          const steps = [
            { kind: "check", status: "running" },
            { kind: "model", status: "pending" },
            { kind: "verify", status: "pending" },
            { kind: "load", status: "pending" },
          ];
          publish({ phase: "checking", startedAt: Date.now(), steps });
          try {
            const seen = await inspect();
            steps[0].status = "complete";
            if (seen.missing !== undefined) {
              steps[1].status = "running";
              publish({ phase: "downloading", resource: activeFileName, completedBytes: 0, totalBytes: undefined, startedAt: Date.now(), steps });
              const model = await ensureModel(controller.signal);
              steps[1].status = "complete";
              steps[2].status = "running";
              publish({ phase: "checking", startedAt: Date.now(), steps });
              if (model.bytes === 0) throw new Error("gigaam: downloaded file is empty");
              steps[2].status = "complete";
            }
            steps[3].status = "running";
            publish({ phase: "loading", startedAt: Date.now(), steps });
            await startWorker();
            steps[3].status = "complete";
            publish({ phase: "ready", steps });
            scheduleIdleRelease();
          } catch (error) {
            steps.forEach((step) => {
              if (step.status === "running") step.status = "failed";
            });
            publish({
              phase: "failed",
              message: error instanceof Error ? error.message : String(error),
              steps,
            });
          }
        })();
        return preparation;
      },
      async cancel() {
        preparation?.catch(() => {});
        preparation = null;
        stopWorker();
        publish({ phase: "cancelled", startedAt: Date.now() });
      },
    },
    async transcribe(input, signal) {
      signal.throwIfAborted();
      if (state.phase === "unprepared" || state.phase === "failed" || state.phase === "cancelled") {
        provider.preparation.prepare();
        await preparation;
        if (state.phase === "failed") throw new Error(`gigaam: provider is not ready (${state.message ?? "unknown"})`);
      }
      if (idleTimer !== null) {
        clearTimeout(idleTimer);
        idleTimer = null;
      }
      if (worker === null) publish({ phase: "waking", startedAt: Date.now(), steps: state.steps });
      const child = await startWorker();
      const id = ++requestId;
      return await new Promise((resolveTranscription, rejectTranscription) => {
        const onAbort = () => {
          child.off("message", onMessage);
          // Cancellation terminates the active worker: the host releases its RAM
          // immediately instead of waiting for the current inference to finish.
          stopWorker();
          rejectTranscription(new Error("gigaam: transcription cancelled"));
        };
        const onMessage = (message) => {
          if (message?.id !== id) return;
          child.off("message", onMessage);
          signal.removeEventListener("abort", onAbort);
          if (message.ok) {
            publish({ phase: "ready", steps: state.steps });
            scheduleIdleRelease();
            resolveTranscription({ text: message.text, audioSeconds: message.audioSeconds, inferenceSeconds: message.inferenceSeconds });
          } else {
            rejectTranscription(new Error(`gigaam: ${message.error}`));
          }
        };
        signal.addEventListener("abort", onAbort, { once: true });
        child.on("message", onMessage);
        child.send({ id, type: "transcribe", wav: Buffer.from(input.audio) }, (error) => {
          if (error !== null && error !== undefined) {
            child.off("message", onMessage);
            signal.removeEventListener("abort", onAbort);
            rejectTranscription(new Error(`gigaam: cannot send the recording (${error.message})`));
          }
        });
      });
    },
  };

  // Activation only inspects cached resources; the worker is woken by the first
  // recording. Synchronous so the first snapshot a consumer sees is already
  // truthful about whether the model is present.
  try {
    const seen = inspect();
    publish(seen.ready ? { phase: "standby", steps: [] } : { phase: "unprepared", steps: [] });
  } catch (error) {
    publish({ phase: "failed", message: error instanceof Error ? error.message : String(error), steps: [] });
  }

  return provider;
}

/**
 * @param {import("@deepseek-ai/cordis").Context} ctx
 * @param {z.infer<typeof Config>} config
 * @returns {() => void} disposer
 */
function apply(ctx, config) {
  const provider = createProvider(ctx, config);
  const dispose = ctx.speechToText.register(provider);
  if (config.setAsDefaultProvider) {
    void ctx.speechToText
      .configure({ providerId: provider.info.id, language: config.language })
      .catch((error) => {
        ctx.logger?.warn?.(`stt-gigaam: could not select itself as the default provider: ${error?.message ?? error}`);
      });
  }
  return dispose;
}

export { apply };
export default { name, inject, Config, apply };
