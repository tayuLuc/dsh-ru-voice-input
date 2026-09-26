/**
 * Pure decoding helpers for the GigaAM v3 CTC provider.
 *
 * No I/O beyond reading the vocabulary file, no runtime dependency on the
 * plugin entry, so the worker and the tests can share exactly one code path.
 */
import { readFileSync } from "node:fs";

/** Canonical recording format accepted by the DeepSeek Harness speech stack. */
export const SAMPLE_RATE = 16000;
const BITS_PER_SAMPLE = 16;
const CHANNELS = 1;

/**
 * Parse a canonical WAV file: RIFF/WAVE, mono, 16-bit PCM.
 *
 * @param {Uint8Array} bytes - whole file content
 * @returns {{ pcm: Float32Array, sampleRate: number, seconds: number }} mono float samples in [-1, 1)
 * @throws {Error} on any header, format, or length inconsistency
 */
export function parseWav(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (offset) => String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
  if (bytes.length < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("gigaam: not a RIFF/WAVE file");
  let offset = 12;
  let format = null;
  let data = null;
  while (offset + 8 <= bytes.length) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    const end = offset + 8 + size;
    if (end > bytes.length) throw new Error(`gigaam: chunk ${id} overruns the file`);
    if (id === "fmt ") {
      format = {
        audioFormat: view.getUint16(offset + 8, true),
        channels: view.getUint16(offset + 10, true),
        sampleRate: view.getUint32(offset + 12, true),
        bitsPerSample: view.getUint16(offset + 22, true),
      };
    } else if (id === "data") {
      data = bytes.subarray(offset + 8, end);
    }
    offset = end + (size & 1);
  }
  if (format === null) throw new Error("gigaam: missing fmt chunk");
  if (data === null) throw new Error("gigaam: missing data chunk");
  if (format.audioFormat !== 1) throw new Error(`gigaam: expected PCM (format 1), got ${format.audioFormat}`);
  if (format.channels !== CHANNELS) throw new Error(`gigaam: expected mono, got ${format.channels} channels`);
  if (format.bitsPerSample !== BITS_PER_SAMPLE) throw new Error(`gigaam: expected 16-bit, got ${format.bitsPerSample}-bit`);
  if (format.sampleRate !== SAMPLE_RATE) throw new Error(`gigaam: expected ${SAMPLE_RATE} Hz, got ${format.sampleRate} Hz`);
  if (data.length % 2 !== 0) throw new Error("gigaam: PCM16 payload has an odd byte count");
  const count = data.length >> 1;
  if (count < 320) throw new Error(`gigaam: recording too short (${count} samples)`);
  // `view` indices are relative to the start of `bytes`, so translate the
  // subarray offset before reading each 16-bit sample.
  const dataStart = data.byteOffset - bytes.byteOffset;
  const pcm = new Float32Array(count);
  for (let i = 0; i < count; i++) pcm[i] = view.getInt16(dataStart + i * 2, true) / 32768;
  return { pcm, sampleRate: format.sampleRate, seconds: count / format.sampleRate };
}

/**
 * Load the character vocabulary: one `token id` pair per line, where the
 * SentencePiece boundary marker U+2581 stands for a space and `<blk>` is the
 * CTC blank.
 *
 * @param {string} path
 * @returns {{ tokens: string[], blankId: number }}
 */
export function loadVocab(path) {
  const tokens = [];
  let blankId = -1;
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (trimmed === "") continue;
    const [token, rawId] = trimmed.split(" ");
    const id = Number(rawId);
    if (!Number.isInteger(id)) throw new Error(`gigaam: vocab line without an id: ${trimmed}`);
    tokens[id] = token === "▁" ? " " : token;
    if (token === "<blk>") blankId = id;
  }
  if (blankId < 0) throw new Error("gigaam: vocabulary has no <blk> token");
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === undefined) throw new Error(`gigaam: vocabulary is missing id ${i}`);
  }
  return { tokens, blankId };
}

/**
 * Greedy CTC decode over log-probabilities laid out as
 * `[frames][vocabSize]` inside one flat array.
 *
 * Mirrors the reference implementation: per-frame argmax, drop the blank,
 * then drop a token that repeats the previous emitted frame.
 *
 * @param {Float32Array} logProbs - flat log-probability tensor
 * @param {number} vocabSize - classes per frame
 * @param {number} frames - usable frames
 * @param {number} blankId - CTC blank class
 * @returns {number[]} emitted token ids
 */
export function greedyCtc(logProbs, vocabSize, frames, blankId) {
  const ids = [];
  let previous = blankId;
  for (let t = 0; t < frames; t++) {
    const base = t * vocabSize;
    let best = 0;
    let bestValue = -Infinity;
    for (let c = 0; c < vocabSize; c++) {
      const value = logProbs[base + c];
      if (value > bestValue) {
        bestValue = value;
        best = c;
      }
    }
    if (best !== blankId && best !== previous) ids.push(best);
    previous = best;
  }
  return ids;
}

const isWordCharacter = (character) => /[\p{L}\p{N}]/u.test(character);

/**
 * Join token ids into a transcript: keep one space only between two word
 * characters, drop leading and repeated spaces, trim the result.
 *
 * @param {number[]} ids - emitted token ids
 * @param {string[]} tokens - id to character map
 * @returns {string}
 */
export function assembleText(ids, tokens) {
  const raw = ids.map((id) => tokens[id] ?? "").join("");
  let out = "";
  let i = 0;
  while (i < raw.length) {
    const character = raw[i];
    if (character !== " ") {
      out += character;
      i += 1;
      continue;
    }
    // Collapse the whole run of spaces, then keep one only when it sits
    // between two word characters.
    let before = i - 1;
    while (before >= 0 && raw[before] === " ") before -= 1;
    let after = i + 1;
    while (after < raw.length && raw[after] === " ") after += 1;
    if (before >= 0 && after < raw.length && isWordCharacter(raw[before]) && isWordCharacter(raw[after])) {
      out += " ";
    }
    i = after;
  }
  return out.trim();
}
