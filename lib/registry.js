/**
 * dsh-ru-speech-to-text — the official speech registry with one behavioural fix.
 *
 * Fork of @deepseek-ai/dsh-experimental-speech-to-text (0.1.7-rc.2). The service
 * name, config schema, and every method contract are unchanged, so the official
 * remote bridge, the microphone UI, and third-party providers all work against
 * this build untouched.
 *
 * The fix is in configure(). Upstream validates the CURRENT language against
 * the NEWLY selected provider:
 *
 *     const id = patch.providerId ?? this.config.defaultProvider.get();
 *     this.selectedProvider(id, patch.language ?? this.config.language.get());
 *
 * That makes a provider switch impossible whenever the stored language is not
 * in the new provider's list — a Russian profile on `ru` cannot select the
 * stock SenseVoice provider, and a profile left on `auto` cannot select a
 * Russian-only one. The setting UI reports the failure by silently reverting
 * the selection, with no usable way out. Here the language follows the provider
 * instead: an explicit language is still validated and still rejected when the
 * provider cannot accept it, while a bare provider switch adopts a language the
 * new provider does accept.
 */
import { Service } from "@deepseek-ai/cordis";
import z from "@deepseek-ai/schemastery";

/** Named transcription providers with disposable registration and explicit routing. */
/** Registry shared by all transcription consumers in one Host composition. */
var SpeechToText = class extends Service {
	config;
	static Config = z.object({
		defaultProvider: z.string().min(1).required().volatile(),
		language: z.string().min(1).default("auto").volatile()
	});
	providers = /* @__PURE__ */ new Map();
	listeners = /* @__PURE__ */ new Set();
	lifetime = new AbortController();
	/** Profile-local entry id used by Settings; absent when the plugin was mounted without Loader. */
	entryId;
	constructor(ctx, config) {
		super(ctx, "speechToText");
		this.config = config;
		this.entryId = ctx.fiber.entry?.options.id;
		ctx.on("loader/volatile-update", () => {
			this.changed();
		});
		ctx.effect(() => async () => {
			this.lifetime.abort(/* @__PURE__ */ new Error("Speech service disposed"));
			await Promise.all([...this.providers.values()].map((registration) => this.remove(registration)));
			this.listeners.clear();
		});
	}
	/**
	* Register one recognizer; duplicate ids fail without replacing the original.
	* @param provider - recognizer owned by the contributing fiber.
	* @returns idempotent disposer which rejects admission, cancels, and joins accepted work.
	*/
	register(provider) {
		this.lifetime.signal.throwIfAborted();
		if (this.providers.has(provider.info.id)) throw new Error(`Speech provider already registered: ${provider.info.id}`);
		const registration = {
			provider,
			lifetime: new AbortController(),
			pending: /* @__PURE__ */ new Set(),
			unsubscribe: provider.preparation?.subscribe(() => {
				this.changed();
			}) ?? (() => {})
		};
		this.providers.set(provider.info.id, registration);
		this.changed();
		return async () => {
			await this.remove(registration);
		};
	}
	async remove(registration) {
		if (this.providers.get(registration.provider.info.id) !== registration) return;
		this.providers.delete(registration.provider.info.id);
		registration.unsubscribe();
		this.changed();
		registration.lifetime.abort(/* @__PURE__ */ new Error("Speech provider unloaded"));
		await Promise.allSettled(registration.pending);
	}
	/**
	* Read the current recognizer roster.
	* @returns available provider facts in registration order.
	*/
	listProviders() {
		return [...this.providers.values()].map(({ provider }) => provider.info);
	}
	changed() {
		for (const listener of this.listeners) listener();
	}
	/**
	* Observe complete readiness snapshots; a slow reader coalesces intermediate progress.
	* @param caller - observer lifetime, independent of any preparation task.
	* @returns an initial snapshot followed by the latest provider states.
	*/
	async *follow(caller) {
		const signal = AbortSignal.any([caller, this.lifetime.signal]);
		signal.throwIfAborted();
		let wake = Promise.withResolvers();
		let changed = true;
		const aborted = () => signal.aborted;
		const notify = () => {
			changed = true;
			wake.resolve(void 0);
		};
		this.listeners.add(notify);
		signal.addEventListener("abort", notify, { once: true });
		try {
			while (!aborted()) {
				if (!changed) await wake.promise;
				if (aborted()) break;
				wake = Promise.withResolvers();
				changed = false;
				yield this.snapshot();
			}
		} finally {
			this.listeners.delete(notify);
			signal.removeEventListener("abort", notify);
		}
	}
	/**
	* Read provider readiness and current user preferences together.
	* @returns one detached complete observation.
	*/
	snapshot() {
		return {
			providers: [...this.providers.values()].map(({ provider }) => ({
				...provider.info,
				preparation: provider.preparation?.snapshot() ?? { phase: "ready" }
			})),
			selection: {
				providerId: this.config.defaultProvider.get(),
				language: this.config.language.get()
			}
		};
	}
	/**
	* Persist changed selection fields into this plugin's profile entry; the resulting language must be accepted by the selected provider.
	*
	* A bare provider switch adopts a language the new provider accepts: the
	* stored one when possible, otherwise the first it advertises. An explicit
	* `language` in the patch is never substituted and still fails loudly.
	* @param patch - explicit provider or language changes.
	* @returns after the profile write and the live update it applies.
	*/
	async configure(patch) {
		const settings = this.ctx.get("settings");
		const entry = this.entryId;
		if (settings === void 0 || entry === void 0) throw new Error("Speech selection requires the settings service and a profile entry");
		const id = patch.providerId ?? this.config.defaultProvider.get();
		const language = this.selectLanguage(id, patch.language ?? this.config.language.get(), patch.language === void 0 && patch.providerId !== void 0);
		this.selectedProvider(id, language);
		await settings.update(entry, {
			...patch.providerId === void 0 ? {} : { defaultProvider: patch.providerId },
			...language === this.config.language.get() ? {} : { language }
		});
	}
	/**
	* Resolve the language to store for one provider.
	*
	* @param id - provider being selected.
	* @param candidate - the requested language, explicit or currently stored.
	* @param maySubstitute - true only for a bare provider switch, where the
	* caller named no language and the stored one is not usable.
	* @returns `candidate` when the provider accepts it, else its first language.
	*/
	selectLanguage(id, candidate, maySubstitute) {
		const registration = this.providers.get(id);
		if (registration === void 0) throw new Error(`Speech provider is unavailable: ${id}`);
		const { languages } = registration.provider.info;
		if (languages.includes(candidate)) return candidate;
		if (!maySubstitute) throw new Error(`Speech provider ${id} does not support language: ${candidate}`);
		const [fallback] = languages;
		if (fallback === void 0) throw new Error(`Speech provider ${id} advertises no languages`);
		return fallback;
	}
	selectedProvider(id, language) {
		const registration = this.providers.get(id);
		if (!registration) throw new Error(`Speech provider is unavailable: ${id}`);
		if (!registration.provider.info.languages.includes(language)) throw new Error(`Speech provider ${id} does not support language: ${language}`);
		return registration.provider;
	}
	/**
	* Start or join provider-owned preparation.
	* @param id - exact registered provider identity.
	* @param options - task-local source selection validated by the provider.
	*/
	prepare(id, options) {
		const registration = this.providers.get(id);
		if (!registration) throw new Error(`Speech provider is unavailable: ${id}`);
		registration.provider.preparation?.prepare(options);
	}
	/**
	* Explicitly cancel provider preparation without tying it to a browser connection.
	* @param id - exact registered provider identity.
	* @returns after the preparation task settles.
	*/
	async cancelPreparation(id) {
		const registration = this.providers.get(id);
		if (!registration) throw new Error(`Speech provider is unavailable: ${id}`);
		await registration.provider.preparation?.cancel();
	}
	/**
	* Apply composition defaults and capture the selected provider. Missing providers and unsupported languages fail explicitly.
	* @param request - complete recording and optional selection.
	* @returns provider-pinned input for transcribe().
	*/
	resolve(request) {
		const id = request.providerId ?? this.config.defaultProvider.get();
		const language = request.language ?? this.config.language.get();
		return {
			provider: this.selectedProvider(id, language),
			audio: request.audio,
			language
		};
	}
	/**
	* Execute exactly the resolved provider; no fallback sends audio elsewhere.
	* @param spec - resolved input, a withdrawn or replaced registration is rejected.
	* @param signal - caller cancellation.
	* @returns final transcript after provider settlement.
	*/
	async transcribe(spec, signal) {
		signal.throwIfAborted();
		const registration = this.providers.get(spec.provider.info.id);
		if (registration?.provider !== spec.provider) throw new Error("Resolved speech provider is no longer registered");
		const combined = AbortSignal.any([signal, registration.lifetime.signal]);
		const pending = Promise.resolve().then(() => {
			combined.throwIfAborted();
			return spec.provider.transcribe({
				audio: spec.audio,
				language: spec.language
			}, combined);
		});
		registration.pending.add(pending);
		try {
			return await pending;
		} finally {
			registration.pending.delete(pending);
		}
	}
};
//#endregion
export { SpeechToText as default };
