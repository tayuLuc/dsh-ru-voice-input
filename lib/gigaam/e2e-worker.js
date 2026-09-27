/**
 * E2E worker for the GigaAM provider: MLX through a Python sidecar.
 *
 * Same wire protocol as the other two workers, so the provider's lifecycle —
 * fork on demand, ping to warm, terminate to release memory — is unchanged.
 * What runs inside is `gigaam-mlx`, the MLX port of GigaAM v3 e2e: the only
 * engine in this family that emits punctuation and capitalisation, because the
 * CTC heads have a 34-token letter vocabulary and cannot produce either.
 *
 * Measured on this Apple M1 host (27.09), warm:
 *   e2e ctc    RTF 0.027 (5 s) / 0.017 (28.6 s)  -> punctuated text
 *   metal gguf RTF 0.016 (28.6 s)                -> no punctuation
 *
 * The sidecar loads the model before it announces readiness, so the parent's
 * ping means "the weights are in memory" and not merely "a process exists".
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseWav } from "./decode.js";

const SIDECAR = fileURLToPath(new URL("./sidecar.py", import.meta.url));
/** Big-endian (request id, payload size), matching the sidecar's struct. */
const HEADER_BYTES = 8;

const options = JSON.parse(process.env.GIGAAM_WORKER_OPTIONS ?? "{}");

/** @type {import("node:child_process").ChildProcessWithoutNullStreams | null} */
let sidecar = null;
let sidecarFailure = null;
/** Resolves once the sidecar reports the model is in memory. */
let ready = null;
const waiting = new Map();
let nextRequest = 1;

function failAll(reason) {
  sidecarFailure = reason;
  for (const [, resolve] of waiting) resolve({ ok: false, error: reason });
  waiting.clear();
}

/** Start the Python sidecar and return a promise that settles on its ready line. */
function ensureSidecar() {
  if (ready !== null) return ready;
  if (sidecarFailure !== null) return Promise.reject(new Error(sidecarFailure));
  const python = options.pythonPath;
  if (typeof python !== "string" || python.length === 0) {
    return Promise.reject(new Error("gigaam e2e: no Python interpreter configured (set e2ePythonPath)"));
  }
  if (!existsSync(python)) {
    return Promise.reject(new Error(`gigaam e2e: Python interpreter not found at ${python}`));
  }
  ready = new Promise((resolve) => {
    const child = spawn(python, [SIDECAR], {
      stdio: ["pipe", "pipe", "inherit"],
      // gigaam-mlx shells out to ffmpeg to decode the recording. A GUI-launched
      // host often has a minimal PATH that omits Homebrew, so the common
      // locations are prepended rather than assuming the operator's shell.
      env: {
        ...process.env,
        PATH: ["/opt/homebrew/bin", "/usr/local/bin", process.env.PATH ?? ""].join(":"),
        GIGAAM_E2E_MODEL: options.e2eModel ?? "ctc"
      },
    });
    sidecar = child;

    let buffer = "";
    child.stdout.on("data", (chunk) => {
      buffer += chunk.toString("utf8");
      let newline = buffer.indexOf("\n");
      while (newline >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line.length > 0) {
          let parsed;
          try {
            parsed = JSON.parse(line);
          } catch {
            parsed = null;
          }
          if (parsed !== null) {
            if (parsed.id === 0) {
              resolve();
            } else {
              const entry = waiting.get(parsed.id);
              if (entry !== undefined) {
                waiting.delete(parsed.id);
                entry(parsed);
              }
            }
          }
        }
        newline = buffer.indexOf("\n");
      }
    });

    child.once("error", (error) => {
      sidecar = null;
      ready = null;
      failAll(`gigaam e2e: cannot start the sidecar: ${error.message}`);
    });
    child.once("exit", (code) => {
      sidecar = null;
      ready = null;
      failAll(`gigaam e2e: sidecar exited with code ${code}`);
    });
  });
  return ready;
}

/** Send one recording and await its JSON reply. */
async function request(wav) {
  const loaded = ensureSidecar();
  const id = nextRequest++;
  const reply = new Promise((resolve) => {
    waiting.set(id, resolve);
  });
  // Writing before the model finishes loading is deliberate: the sidecar reads
  // frames only after it is warm, so the request simply waits in its pipe.
  const stream = sidecar.stdin;
  const header = Buffer.alloc(HEADER_BYTES);
  header.writeUInt32BE(id, 0);
  header.writeUInt32BE(wav.length, 4);
  stream.write(header);
  stream.write(wav);
  return loaded.then(() => reply);
}

/** Accept both IPC encodings: advanced keeps a view, JSON gives a plain object. */
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
        await ensureSidecar();
        process.send({ id, ok: true, ready: true, deviceType: "gpu", device: "MLX" });
      } else if (type === "transcribe") {
        const wav = toBytes(message.wav);
        parseWav(wav); // reject malformed audio before spending a round trip
        const reply = await request(wav);
        if (reply.ok === true) {
          process.send({
            id,
            ok: true,
            text: reply.text,
            audioSeconds: reply.audioSeconds,
            inferenceSeconds: reply.inferenceSeconds
          });
        } else {
          process.send({ id, ok: false, error: String(reply.error) });
        }
      } else if (type === "shutdown") {
        sidecar?.kill("SIGKILL");
        process.exit(0);
      }
    } catch (error) {
      process.send({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
    }
  })();
});

process.on("uncaughtException", (error) => {
  process.send({ id: 0, ok: false, error: `gigaam e2e worker crashed: ${error.message}` });
  process.exit(1);
});
