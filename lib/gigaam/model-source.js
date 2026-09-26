/**
 * Pinned model acquisition for the GigaAM provider.
 *
 * The heavy artefact (int8 CTC encoder) is fetched from a revision-pinned
 * Hugging Face repository and verified by SHA-256 before it is accepted. The
 * small feature-extraction graph and the vocabulary ship inside the package, so
 * a broken download can never produce a half-usable provider: verification
 * fails closed and the caller keeps the previous state.
 */
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

/** Default Hugging Face repository holding the GigaAM v3 ONNX exports. */
export const DEFAULT_REPO = "istupakov/gigaam-v3-onnx";
/** Pinned revision: verified on 2026-09-26, file `v3_ctc.int8.onnx` = 224.7 MB. */
export const DEFAULT_REVISION = "322c3b29492673eb7d0b434bfa9dfb8653e34d02";
export const DEFAULT_MODEL_FILE = "v3_ctc.int8.onnx";
export const DEFAULT_VOCAB_FILE = "v3_vocab.txt";
export const DEFAULT_MODEL_SHA256 = "ceb61454e2e1a2dec5872cbac1de0fe0a4271d1148f6b26b5bda53ff30a12acd";
export const DEFAULT_VOCAB_SHA256 = "a9143c30844d3c0bee3e9e927e4084774eb1b9eeaafc473b2c4521e4911a7c07";

/**
 * @param {string} path
 * @returns {Promise<string>} lowercase hex SHA-256 of the file
 */
export async function sha256File(path) {
  const hash = createHash("sha256");
  await pipeline(createReadStream(path), hash);
  return hash.digest("hex");
}

/**
 * @param {string} path
 * @returns {Promise<number | undefined>} file size in bytes, undefined when absent
 */
export async function fileSize(path) {
  try {
    return (await stat(path)).size;
  } catch {
    return undefined;
  }
}

/** One managed download step, mirroring the Host preparation vocabulary. */
export class ModelSource {
  /**
   * @param {object} options
   * @param {string} options.directory - managed directory for the model files
   * @param {string} [options.origin] - download origin, Hugging Face compatible
   * @param {string} [options.repo] - repository id
   * @param {string} [options.revision] - pinned revision
   * @param {(event: { file: string, completedBytes: number, totalBytes?: number }) => void} [options.onProgress]
   * @param {number} [options.timeoutMs] - per-request timeout
   */
  constructor(options) {
    this.directory = options.directory;
    this.origin = (options.origin ?? "https://huggingface.co").replace(/\/+$/, "");
    this.repo = options.repo ?? DEFAULT_REPO;
    this.revision = options.revision ?? DEFAULT_REVISION;
    this.onProgress = options.onProgress ?? (() => {});
    this.timeoutMs = options.timeoutMs ?? 600_000;
  }

  /** Absolute path of one managed file. */
  path(file) {
    return join(this.directory, file);
  }

  /** Public download URL for one pinned file. */
  url(file) {
    return `${this.origin}/${this.repo}/resolve/${this.revision}/${file}`;
  }

  /**
   * Ensure one file exists with the expected digest.
   *
   * @param {string} file - file name inside the managed directory
   * @param {string} expectedSha256 - lowercase hex digest
   * @param {AbortSignal} [signal]
   * @returns {Promise<{ path: string, bytes: number, restored: boolean }>}
   */
  async ensure(file, expectedSha256, signal) {
    const target = this.path(file);
    const existingBytes = await fileSize(target);
    if (existingBytes !== undefined) {
      const digest = await sha256File(target);
      if (digest === expectedSha256) return { path: target, bytes: existingBytes, restored: true };
      throw new Error(
        `gigaam: ${file} is present but its digest is ${digest.slice(0, 12)}…, expected ${expectedSha256.slice(0, 12)}…; ` +
          "remove the file to re-download it",
      );
    }
    await mkdir(this.directory, { recursive: true });
    const partial = `${target}.partial`;
    const url = this.url(file);
    let response;
    try {
      response = await fetch(url, { signal, redirect: "follow" });
    } catch (cause) {
      throw new Error(`gigaam: cannot reach ${this.origin} for ${file}: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
    }
    if (!response.ok || response.body === null) {
      throw new Error(`gigaam: download of ${file} failed with HTTP ${response.status}`);
    }
    const total = Number(response.headers.get("content-length") ?? "0") || undefined;
    const counter = { completedBytes: 0 };
    const source = Readable.fromWeb(response.body);
    source.on("data", (chunk) => {
      counter.completedBytes += chunk.length;
      this.onProgress({ file, completedBytes: counter.completedBytes, totalBytes: total });
    });
    try {
      await pipeline(source, createWriteStream(partial), { signal });
    } catch (cause) {
      await rm(partial, { force: true });
      throw new Error(`gigaam: download of ${file} did not complete: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
    }
    const digest = await sha256File(partial);
    if (digest !== expectedSha256) {
      await rm(partial, { force: true });
      throw new Error(`gigaam: ${file} digest ${digest.slice(0, 12)}… does not match the pinned ${expectedSha256.slice(0, 12)}…`);
    }
    await rename(partial, target);
    return { path: target, bytes: (await fileSize(target)) ?? 0, restored: false };
  }
}
