/**
 * Behavioural test for the provider-switch fix in dsh-ru-speech-to-text.
 *
 * Upstream validates the CURRENT language against the NEWLY selected provider:
 *
 *     this.selectedProvider(id, patch.language ?? this.config.language.get());
 *
 * so a switch is impossible whenever the stored language is outside the new
 * provider's list. The settings UI reports that by silently reverting the
 * selection. These checks pin the fixed behaviour in both directions, and pin
 * the failures that must stay loud.
 *
 * The real shipped source is loaded, with only its two package imports swapped
 * for local stubs, so these assertions run against the implementation rather
 * than a restatement of it.
 *
 * Run with `npm test`.
 */
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

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

const here = fileURLToPath(new URL(".", import.meta.url));

/**
 * Load the real registry source with its package imports replaced by stubs.
 * Nothing else in the file is altered.
 */
async function loadRegistry() {
	const source = await readFile(join(here, "../lib/registry.js"), "utf8");
	const patched = source
		.replace('from "@deepseek-ai/cordis"', `from ${JSON.stringify(pathToFileURL(join(here, "stubs/cordis.mjs")).href)}`)
		.replace('from "@deepseek-ai/schemastery"', `from ${JSON.stringify(pathToFileURL(join(here, "stubs/schemastery.mjs")).href)}`);
	const directory = await mkdtemp(join(tmpdir(), "ru-stt-"));
	const file = join(directory, "registry.mjs");
	await writeFile(file, patched, "utf8");
	const module = await import(pathToFileURL(file).href);
	return module.default;
}

const SpeechToText = await loadRegistry();

/** A provider stub carrying only what the registry reads. */
function provider(id, languages) {
	return {
		info: { id, name: id, location: "host-local", languages },
		transcribe: async () => ({ text: "", audioSeconds: 0, inferenceSeconds: 0 })
	};
}

const SENSEVOICE = provider("sensevoice-local", ["auto", "zh", "en", "yue", "ja", "ko"]);
/** The real GigaAM provider also advertises "auto" as an alias for Russian. */
const GIGAAM = provider("gigaam-v3-ctc-local", ["ru", "auto"]);
/** A provider that accepts Russian only — the discriminating case for the fix. */
const RU_ONLY = provider("ru-only-local", ["ru"]);

/**
 * Build a live registry instance with an in-memory settings service.
 *
 * The stored selection is keyed the way the host keys it (`defaultProvider`),
 * and `settings.update` applies the same patch to the live config, so a write
 * is visible to the next read exactly as it is in the real service.
 * @returns the registry, its live selection state, and the recorded writes.
 */
function mount(providers, stored) {
	const state = { ...stored };
	const written = [];
	const settings = {
		async update(entry, patch) {
			written.push({ entry, patch });
			Object.assign(state, patch);
		}
	};
	const config = {
		defaultProvider: { get: () => state.defaultProvider ?? state.providerId },
		language: { get: () => state.language }
	};
	// The constructor subscribes to loader events and registers a disposal
	// effect; both are inert here but must exist.
	const ctx = {
		fiber: { entry: { options: { id: "speech-to-text" } } },
		on: () => {},
		effect: () => {},
		get: (name) => (name === "settings" ? settings : undefined)
	};
	const service = new SpeechToText(ctx, config);
	for (const p of providers) service.register(p);
	return { service, state, written, config, providerId: () => state.defaultProvider ?? state.providerId };
}

await check("a bare switch to a Russian-only provider from language=auto adopts Russian", async () => {
	const { service, state, providerId } = mount([SENSEVOICE, RU_ONLY], { providerId: "sensevoice-local", language: "auto" });
	await service.configure({ providerId: "ru-only-local" });
	assert.equal(providerId(), "ru-only-local");
	assert.equal(state.language, "ru");
});

await check("a bare switch to GigaAM keeps auto, which it accepts as Russian", async () => {
	const { service, state, providerId } = mount([SENSEVOICE, GIGAAM], { providerId: "sensevoice-local", language: "auto" });
	await service.configure({ providerId: "gigaam-v3-ctc-local" });
	assert.equal(providerId(), "gigaam-v3-ctc-local");
	assert.equal(state.language, "auto");
});

await check("a bare switch back to the stock provider from ru succeeds", async () => {
	const { service, state, providerId } = mount([SENSEVOICE, GIGAAM], { providerId: "gigaam-v3-ctc-local", language: "ru" });
	await service.configure({ providerId: "sensevoice-local" });
	assert.equal(providerId(), "sensevoice-local");
	assert.equal(state.language, "auto");
});

await check("a stored language the new provider accepts is kept", async () => {
	const { service, state, providerId } = mount([SENSEVOICE, GIGAAM], { providerId: "gigaam-v3-ctc-local", language: "auto" });
	await service.configure({ providerId: "gigaam-v3-ctc-local" });
	assert.equal(state.language, "auto");
});

await check("an explicit unsupported language is still rejected", async () => {
	const { service } = mount([SENSEVOICE, GIGAAM], { providerId: "sensevoice-local", language: "auto" });
	await assert.rejects(() => service.configure({ providerId: "gigaam-v3-ctc-local", language: "zh" }), /does not support language: zh/);
});

await check("an unknown provider still fails loudly", async () => {
	const { service } = mount([SENSEVOICE, GIGAAM], { providerId: "sensevoice-local", language: "auto" });
	await assert.rejects(() => service.configure({ providerId: "no-such-provider" }), /Speech provider is unavailable: no-such-provider/);
});

await check("a language-only change is validated against the current provider", async () => {
	const { service } = mount([SENSEVOICE, GIGAAM], { providerId: "gigaam-v3-ctc-local", language: "ru" });
	await assert.rejects(() => service.configure({ language: "zh" }), /does not support language: zh/);
});

await check("a language-only change to a supported language is stored", async () => {
	const { service, state, providerId } = mount([SENSEVOICE, GIGAAM], { providerId: "gigaam-v3-ctc-local", language: "ru" });
	await service.configure({ language: "auto" });
	assert.equal(state.language, "auto");
	assert.equal(providerId(), "gigaam-v3-ctc-local");
});

await check("resolve() honours the stored selection", async () => {
	const { service } = mount([SENSEVOICE, GIGAAM], { providerId: "gigaam-v3-ctc-local", language: "ru" });
	const spec = service.resolve({ audio: new Uint8Array(64) });
	assert.equal(spec.provider.info.id, "gigaam-v3-ctc-local");
	assert.equal(spec.language, "ru");
});

await check("duplicate provider registration is rejected", async () => {
	const { service } = mount([GIGAAM], { providerId: "gigaam-v3-ctc-local", language: "ru" });
	assert.throws(() => service.register(GIGAAM), /already registered/);
});

await check("transcribe() runs exactly the resolved provider", async () => {
	const seen = [];
	const loud = {
		...GIGAAM,
		transcribe: async (input) => {
			seen.push(input.language);
			return { text: "привет", audioSeconds: 1, inferenceSeconds: 0.1 };
		}
	};
	const { service } = mount([SENSEVOICE, loud], { providerId: "gigaam-v3-ctc-local", language: "ru" });
	const spec = service.resolve({ audio: new Uint8Array(64) });
	const result = await service.transcribe(spec, new AbortController().signal);
	assert.equal(result.text, "привет");
	assert.deepEqual(seen, ["ru"]);
});

if (failures > 0) {
	console.log(`${failures} check(s) failed`);
	process.exit(1);
}
console.log("all checks passed");
