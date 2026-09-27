/**
 * Behavioural checks for engine warmth.
 *
 * The complaint these cover is not a wrong transcript but a *late* one: the
 * worker is released after an idle window, so the next recording pays a reload.
 * That is invisible in a transcript comparison, so the assertions are on
 * readiness and on the promise that warming never downloads anything.
 *
 * The engine is picked from what this machine actually has, because the test
 * warms a real model rather than a mock: a mock cannot show the load finished.
 *
 * Run with: node test/warmth.mjs [modelDirectory]
 */
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
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
console.log(`прогреваем движок: ${chosen.engine} (${chosen.why})\n`);

const { apply } = await import(new URL("../lib/gigaam/index.js", import.meta.url));

function mount(overrides = {}) {
	const registrations = [];
	const warnings = [];
	const ctx = {
		speechToText: {
			register(provider) {
				registrations.push(provider);
				return () => registrations.splice(registrations.indexOf(provider), 1);
			},
			async configure() {}
		},
		logger: { warn: (message) => warnings.push(message) }
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
		...overrides
	};
	const dispose = apply(ctx, config);
	return { provider: registrations[0], dispose, warnings, config };
}

const waitFor = async (provider, wanted, attempts = 200) => {
	let phase = provider.preparation.snapshot().phase;
	for (let attempt = 0; attempt < attempts && phase !== wanted; attempt += 1) {
		await new Promise((resolve) => setTimeout(resolve, 250));
		phase = provider.preparation.snapshot().phase;
	}
	return phase;
};

await check("без прогрева движок не поднимается сам и ждёт записи", async () => {
	const { provider, dispose } = mount({ preload: false });
	assert.equal(provider.preparation.snapshot().phase, "standby");
	await new Promise((resolve) => setTimeout(resolve, 600));
	assert.equal(provider.preparation.snapshot().phase, "standby");
	await provider.preparation.cancel();
	await dispose();
});

await check("с прогревом движок готов без единой записи", async () => {
	const { provider, dispose, warnings } = mount({ preload: true });
	const phase = await waitFor(provider, "ready");
	assert.equal(phase, "ready", `дошёл до "${phase}" вместо ready; предупреждения: ${warnings.join("; ")}`);
	await provider.preparation.cancel();
	await dispose();
});

await check("прогрев при отсутствующем рантайме не бросает и не заявляет готовность", async () => {
	const { provider, dispose } = mount({
		preload: true,
		engine: "mlx",
		e2ePythonPath: "/nonexistent/python-for-the-warmth-test"
	});
	await new Promise((resolve) => setTimeout(resolve, 800));
	assert.notEqual(provider.preparation.snapshot().phase, "ready", "не должен заявлять готовность без рантайма");
	await provider.preparation.cancel();
	await dispose();
});

if (failures > 0) {
	console.log(`${failures} check(s) failed`);
	process.exit(1);
}
console.log("all checks passed");
