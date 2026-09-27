/**
 * Model cache report and cleanup.
 *
 * The engines share one machine but not one weight file: MLX and onnx-asr keep
 * theirs in the Hugging Face cache, the GGUF and ONNX Runtime engines keep theirs
 * under <dsh home>/models. Only the selected engine's weights are needed, so
 * this reports what is present, what the current `engine` setting actually uses,
 * and how much space everything would free.
 *
 * Usage:
 *   node scripts/models.mjs            # report only, changes nothing
 *   node scripts/models.mjs --prune    # delete the weight sets the engine is not using
 *   node scripts/models.mjs --prune --engine mlx   # ...assuming that engine is the one in use
 */
import { execFileSync } from "node:child_process";
import { existsSync, lstatSync, readdirSync, realpathSync, rmSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join, sep } from "node:path";

const args = new Set(process.argv.slice(2));
const prune = args.has("--prune");
const engineArg = process.argv.includes("--engine") ? process.argv[process.argv.indexOf("--engine") + 1] : null;

const dshHome = process.env.DSH_HOME ?? join(homedir(), ".dsh");
const hfHome = process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface");

/** The engine the provider would resolve to on this machine. */
function autoEngine() {
  if (process.platform === "darwin" && process.arch === "arm64") return "mlx";
  return "onnx-asr";
}

/**
 * Resolve the newest file under a Hugging Face snapshot directory.
 *
 * The snapshot entries are symlinks into a shared blob store, so the store
 * cannot be measured as a whole without counting the same bytes once per
 * repository. Measuring the individual weight files keeps the numbers honest.
 */
function newestSnapshotFile(repoDir, fileName) {
  const snapshots = join(hfHome, "hub", repoDir, "snapshots");
  if (!existsSync(snapshots)) return undefined;
  const candidates = readdirSync(snapshots)
    .map((revision) => join(snapshots, revision, fileName))
    .filter((candidate) => existsSync(candidate))
    .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
  return candidates[0];
}

/**
 * Every weight file this plugin can use, and which engine owns it.
 * The active engine is the one `auto` resolves to unless overridden.
 */
const SETS = [
  {
    id: "mlx",
    note: "MLX e2e — пунктуация; дефолт на Apple Silicon",
    resolve: () => newestSnapshotFile("models--aystream--GigaAM-v3-e2e-ctc-mlx", "weights.safetensors")
  },
  {
    id: "onnx-asr",
    note: "ONNX e2e — пунктуация; портретный движок, умеет CUDA",
    resolve: () => newestSnapshotFile("models--istupakov--gigaam-v3-onnx", "v3_e2e_ctc.onnx")
  },
  {
    id: "gguf",
    note: "GGUF Q8_0 через transcribe.cpp — без пунктуации",
    resolve: () => {
      const path = join(dshHome, "models", "gigaam-v3-ctc", "gigaam-v3-ctc-Q8_0.gguf");
      return existsSync(path) ? path : undefined;
    }
  },
  {
    id: "onnx",
    note: "ONNX int8 CTC — без пунктуации, без Python",
    resolve: () => {
      const path = join(dshHome, "models", "gigaam-v3-ctc", "v3_ctc.int8.onnx");
      return existsSync(path) ? path : undefined;
    }
  }
];

function directorySize(path) {
  try {
    // `du -L` follows the symlinks a Hugging Face snapshot is built from.
    return Number(execFileSync("du", ["-skL", path], { encoding: "utf8" }).split("\t")[0]) * 1024;
  } catch {
    return 0;
  }
}

function freeBytes(path) {
  try {
    return Number(execFileSync("df", ["-k", path], { encoding: "utf8" }).trim().split("\n").pop().split(/\s+/)[3]) * 1024;
  } catch {
    return null;
  }
}

const megabytes = (bytes) => `${(bytes / 1048576).toFixed(0)} МБ`;
const gigabytes = (bytes) => `${(bytes / 1073741824).toFixed(2)} ГБ`;

const active = engineArg ?? autoEngine();
const keep = new Set([active]);

let reclaimable = 0;
let present = 0;
console.log(`движок в силе: ${active}\n`);
for (const set of SETS) {
  const path = set.resolve();
  if (path === undefined) {
    console.log(`  [нет]        ${set.id.padEnd(9)} ${set.note}`);
    continue;
  }
  const size = statSync(path).size;
  present += 1;
  if (keep.has(set.id)) {
    console.log(`  [используется] ${set.id.padEnd(9)} ${megabytes(size).padStart(8)}  ${set.note}`);
  } else {
    reclaimable += size;
    console.log(`  [кандидат]     ${set.id.padEnd(9)} ${megabytes(size).padStart(8)}  ${set.note}`);
  }
}

console.log(`\nнаборов на диске: ${present}`);
if (reclaimable > 0) {
  console.log(`не используются активным движком: ${gigabytes(reclaimable)}`);
}
/**
 * Delete blob-store entries that no snapshot points at any more.
 *
 * A Hugging Face snapshot is a symlink into a shared blob store. Removing a
 * link without its blob leaves the bytes on disk forever, which is exactly the
 * slow accumulation this script exists to catch. Anything in blobs/ that no
 * snapshot references is, by definition, unreferenced.
 */
function sweepOrphanBlobs() {
  const blobs = join(hfHome, "hub", "blobs");
  if (!existsSync(blobs)) return { freed: 0, count: 0 };
  const referenced = new Set();
  for (const repo of readdirSync(join(hfHome, "hub"))) {
    const snapshots = join(hfHome, "hub", repo, "snapshots");
    if (!existsSync(snapshots)) continue;
    for (const revision of readdirSync(snapshots)) {
      const dir = join(snapshots, revision);
      for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        try {
          if (existsSync(full) && !statSync(full).isDirectory() && lstatSync(full).isSymbolicLink()) {
            referenced.add(realpathSync(full));
          }
        } catch {
          /* a dangling link references nothing */
        }
      }
    }
  }
  let freed = 0;
  let count = 0;
  for (const prefix of readdirSync(blobs)) {
    const dir = join(blobs, prefix);
    if (!statSync(dir).isDirectory()) continue;
    for (const blob of readdirSync(dir)) {
      const full = join(dir, blob);
      if (referenced.has(full)) continue;
      try {
        freed += statSync(full).size;
        rmSync(full, { force: true });
        count += 1;
      } catch {
        /* raced with another process; leave it */
      }
    }
  }
  return { freed, count };
}

const free = freeBytes(dshHome);
if (free !== null) console.log(`свободно в томе с ${dshHome}: ${gigabytes(free)}`);

if (prune) {
  const orphans = sweepOrphanBlobs();
  if (orphans.count > 0) {
    console.log(`осиротевших блобов удалено: ${orphans.count} (${megabytes(orphans.freed)})`);
  }
}

if (!prune) {
  console.log("\nЭто только отчёт. Убрать лишнее: node scripts/models.mjs --prune");
  process.exit(0);
}
if (reclaimable === 0) {
  console.log("\nУдалять нечего.");
  process.exit(0);
}

console.log("\nУдаляю наборы, которые активный движок не использует...");
for (const set of SETS) {
  if (keep.has(set.id)) continue;
  const path = set.resolve();
  if (path === undefined) continue;
  // A snapshot entry is a symlink into a shared blob store. The real
  // path has to be resolved *before* the link goes, or nothing is left to
  // resolve and the bytes survive as an orphaned blob.
  let blob;
  try {
    const real = realpathSync(path);
    if (real.includes(`${sep}blobs${sep}`)) blob = real;
  } catch {
    blob = undefined;
  }
  rmSync(path, { force: true });
  if (blob !== undefined) rmSync(blob, { force: true });
  console.log(`  удалён ${set.id}`);
}
const after = freeBytes(dshHome);
if (free !== null && after !== null) console.log(`свободно стало: ${gigabytes(after)}`);
console.log("Вес выбранного движка не тронут; остальные докачаются при переключении.");
