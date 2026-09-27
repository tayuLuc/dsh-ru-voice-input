/**
 * Parity between our speech registry fork and the stock one.
 *
 * The fork exists for exactly one behaviour: a bare provider switch must be able
 * to adopt a language the new provider accepts. Everything else must behave
 * identically to upstream, so that the fork can be dropped the day upstream fixes
 * it, and so that a future upstream change is visible as a test failure instead
 * of a surprise at the microphone.
 *
 * Both registries are loaded and driven through the same scenario matrix, and
 * the outcomes are compared. Exactly one difference is expected; anything else
 * is a defect.
 *
 * Run with: node test/parity.mjs
 */
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const stockPath = join(
  homedir(),
  ".dsh",
  "profiles",
  "web",
  "node_modules",
  "@deepseek-ai",
  "dsh-experimental-speech-to-text",
  "lib",
  "index.js"
);
if (!existsSync(stockPath)) {
  console.log(`штатный реестр не найден по ${stockPath}; сравнивать не с чем.`);
  process.exit(2);
}

/**
 * Load a registry source with its two package imports stubbed.
 *
 * `@deepseek-ai/cordis` is supplied by the DSH runtime and is not resolvable from
 * this package, so both sides get the same treatment. Using identical stubs also
 * makes the comparison fair: neither side is loaded with more help than the other.
 */
async function loadRegistry(sourcePath) {
  const source = (await readFile(sourcePath, "utf8"))
    .replace(
      `from ${JSON.stringify("@deepseek-ai/cordis")}`,
      `from ${JSON.stringify(pathToFileURL(join(here, "stubs/cordis.mjs")).href)}`
    )
    .replace(
      `from ${JSON.stringify("@deepseek-ai/schemastery")}`,
      `from ${JSON.stringify(pathToFileURL(join(here, "stubs/schemastery.mjs")).href)}`
    );
  const directory = await mkdtemp(join(tmpdir(), "parity-"));
  const file = join(directory, "registry.mjs");
  await writeFile(file, source, "utf8");
  return (await import(pathToFileURL(file).href)).default;
}

const here = fileURLToPath(new URL(".", import.meta.url));
const Stock = await loadRegistry(stockPath);
const Fork = await loadRegistry(join(here, "..", "lib", "registry.js"));

const SENSEVOICE = { info: { id: "sensevoice-local", location: "host-local", languages: ["auto", "zh"] }, transcribe: async () => ({ text: "" }) };
const RUSSIAN = { info: { id: "ru-only-local", location: "host-local", languages: ["ru"] }, transcribe: async () => ({ text: "" }) };

/** Drive one registry through a scenario and return the outcome as plain data. */
async function run(Registry, scenario) {
  const state = { ...scenario.stored };
  const ctx = {
    fiber: { entry: { options: { id: "speech-to-text" } } },
    on: () => {},
    effect: () => {},
    get: (name) => (name === "settings" ? { async update(_entry, patch) { Object.assign(state, patch); } } : undefined)
  };
  const service = new Registry(ctx, {
    defaultProvider: { get: () => state.providerId },
    language: { get: () => state.language }
  });
  const outcome = { state: {} };
  try {
    for (const p of scenario.providers) service.register(p);
    if (scenario.run) {
      await service.configure(scenario.run);
    } else {
      const spec = service.resolve(scenario.resolve ?? {});
      outcome.resolved = { id: spec.provider.info.id, language: spec.language };
    }
    outcome.state = { ...state };
  } catch (error) {
    outcome.error = error instanceof Error ? error.message : String(error);
  }
  return outcome;
}

const SCENARIOS = [
  { name: "explicit provider and language are honoured", stored: { providerId: "sensevoice-local", language: "auto" }, providers: [SENSEVOICE], run: { providerId: "sensevoice-local", language: "zh" } },
  { name: "explicit unsupported language is rejected", stored: { providerId: "sensevoice-local", language: "auto" }, providers: [SENSEVOICE, RUSSIAN], run: { providerId: "ru-only-local", language: "zh" } },
  { name: "an unknown provider is rejected", stored: { providerId: "sensevoice-local", language: "auto" }, providers: [SENSEVOICE], run: { providerId: "nope" } },
  { name: "a language-only change is validated", stored: { providerId: "ru-only-local", language: "ru" }, providers: [RUSSIAN], run: { language: "zh" } },
  { name: "a compatible language-only change is stored", stored: { providerId: "ru-only-local", language: "ru" }, providers: [RUSSIAN], run: { language: "ru" } },
  { name: "resolve uses the stored selection", stored: { providerId: "ru-only-local", language: "ru" }, providers: [SENSEVOICE, RUSSIAN], resolve: {} },
  { name: "resolve honours explicit request fields", stored: { providerId: "ru-only-local", language: "ru" }, providers: [SENSEVOICE, RUSSIAN], resolve: { providerId: "sensevoice-local", language: "auto" } },
  { name: "resolve rejects an incompatible stored language", stored: { providerId: "ru-only-local", language: "ru" }, providers: [SENSEVOICE], resolve: {} },
  // The single intended difference.
  { name: "bare switch to a Russian-only provider", stored: { providerId: "sensevoice-local", language: "auto" }, providers: [SENSEVOICE, RUSSIAN], run: { providerId: "ru-only-local" }, differs: true },
  { name: "bare switch back to the stock provider", stored: { providerId: "ru-only-local", language: "ru" }, providers: [SENSEVOICE, RUSSIAN], run: { providerId: "sensevoice-local" }, differs: true }
];

let failures = 0;
let differences = 0;

for (const scenario of SCENARIOS) {
  const [stock, fork] = [await run(Stock, scenario), await run(Fork, scenario)];
  const same = JSON.stringify(stock) === JSON.stringify(fork);
  if (scenario.differs) {
    if (same) {
      failures += 1;
      console.log(`FAIL ${scenario.name}: форк НЕ отличается от штатного — правка потеряла смысл`);
    } else {
      differences += 1;
      console.log(`ok   ${scenario.name}: штатный → ${stock.error ?? JSON.stringify(stock.state)}, форк → ${fork.error ?? JSON.stringify(fork.state)}`);
    }
  } else if (same) {
    console.log(`ok   ${scenario.name}: совпадает`);
  } else {
    failures += 1;
    console.log(`FAIL ${scenario.name}: РАСХОЖДЕНИЕ вне заявленной правки`);
    console.log(`     штатный: ${JSON.stringify(stock)}`);
    console.log(`     форк:    ${JSON.stringify(fork)}`);
  }
}

console.log(`\nсценариев: ${SCENARIOS.length}, различий: ${differences}, расхождений: ${failures}`);
if (failures > 0) {
  process.exit(1);
}
console.log("форк ведёт себя как штатный ровно в заявленных границах");
