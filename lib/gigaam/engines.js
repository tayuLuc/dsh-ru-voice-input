/**
 * The engine matrix: what each engine can and cannot do on this machine.
 *
 * One source of truth for three consumers: the provider schema (which engines
 * the settings form offers, and the description under it), the model report
 * script, and the README table. Keeping it as data is what stops the settings
 * from quietly disagreeing with the documentation.
 *
 * `rtf` is measured on this host and null where the combination has not been
 * run here. A null is deliberate: a blank cell is information, a guess is not.
 */

/** Measured on this Apple M1 host, 27.09, warm, Russian. */
export const ENGINES = [
  {
    id: "auto",
    label: "Автовыбор",
    punctuation: true,
    requires: "ничего сверх выбранного по умолчанию",
    diskBytes: 0,
    rtf5s: null,
    rtf28s: null,
    note: "MLX на Apple Silicon, onnx-asr на остальном. Оба ставят пунктуацию."
  },
  {
    id: "mlx",
    label: "MLX e2e — быстрый на Apple",
    punctuation: true,
    requires: "Apple Silicon; Python с gigaam-mlx",
    diskBytes: 843 * 1024 * 1024,
    rtf5s: 0.027,
    rtf28s: 0.017,
    note: "Самая быстрая ветка на M-серии. Первый ответ медленный: компиляция пайплайна."
  },
  {
    id: "onnx-asr",
    label: "ONNX e2e — портретный, с пунктуацией",
    punctuation: true,
    requires: "Python с onnx-asr и onnxruntime",
    diskBytes: 845 * 1024 * 1024,
    rtf5s: 0.048,
    rtf28s: null,
    note: "Единственная ветка с пунктуацией, работающая везде. С CUDA тянет GPU."
  },
  {
    id: "gguf",
    label: "GGUF Q8_0 — быстрый, без пунктуации",
    punctuation: false,
    requires: "ничего, transcribe-cpp в комплекте",
    diskBytes: 272 * 1024 * 1024,
    rtf5s: null,
    rtf28s: 0.016,
    note: "Metal или CUDA. Без знаков: у CTC-словаря 34 токена. Первый запуск на Metal может занять секунды."
  },
  {
    id: "onnx",
    label: "ONNX int8 — минимальный, без пунктуации",
    punctuation: false,
    requires: "ничего сверх профиля",
    diskBytes: 225 * 1024 * 1024,
    rtf5s: 0.038,
    rtf28s: 0.043,
    note: "Единственная ветка, работающая вообще без Python. Запасной вариант."
  }
];

/** Can this engine run here at all, given what the machine offers? */
export function availableEngines(platform = process.platform, arch = process.arch, pythonPresent = true) {
  return ENGINES.filter((engine) => {
    if (engine.id === "auto") return true;
    if (engine.id === "mlx") return platform === "darwin" && arch === "arm64" && pythonPresent;
    if (engine.id === "onnx-asr") return pythonPresent;
    return true;
  }).map((engine) => engine.id);
}

/** The capability table the settings form shows under the engine field. */
export function describeEngines(platform = process.platform, arch = process.arch) {
  const available = new Set(availableEngines(platform, arch));
  const rows = ENGINES
    .filter((engine) => engine.id !== "auto")
    .map((engine) => {
      const parts = [
        engine.label,
        engine.punctuation ? "пунктуация: да" : "пунктуация: нет",
        available.has(engine.id) ? `вес ${mib(engine.diskBytes)}` : "недоступен на этой машине",
        engine.rtf5s === null && engine.rtf28s === null ? "RTF не измерен" : `RTF ${rtfLabel(engine)}`
      ];
      return `- ${parts.join(" · ")}. ${engine.note} Требуется: ${engine.requires}.`;
    });
  return ["Что выбрать:", ...rows].join("\n");
}

function mib(bytes) {
  if (bytes === 0) return "—";
  return `${Math.round(bytes / 1048576)} МБ`;
}

function rtfLabel(engine) {
  if (engine.rtf5s !== null && engine.rtf28s !== null) return `${engine.rtf5s} / ${engine.rtf28s} на 5 и 28.6 с`;
  if (engine.rtf5s !== null) return `${engine.rtf5s} на 5 с`;
  return `${engine.rtf28s} на 28.6 с`;
}

/**
 * Models reachable by name through the sidecar engines, with what each costs.
 *
 * `ortModel` and `e2eModel` are plain strings, so onnx-asr and gigaam-mlx both
 * accept any Hugging Face repo id in those ecosystems. This table is the short
 * list worth showing: it is what a person dictating Russian is likely to
 * consider, with the honest trade-offs attached.
 *
 * `rtf` is measured on this host. `wer` is the vendor's own published figure
 * for the corpus named in `werSource`, not a measurement of ours; the only
 * Russian number we produced ourselves is GigaAM v3 CTC at 7.46 % micro-WER on
 * 491 spontaneous clips, and that was the no-punctuation head.
 */
export const MODELS = [
  {
    id: "gigaam-v3-e2e-ctc",
    where: "onnx-asr",
    languages: "ru, en",
    punctuation: true,
    diskMib: 845,
    wer: null,
    werSource: null,
    note: "Наш дефолт. Русский с пунктуацией и заглавными; самая быстрая на Apple через MLX."
  },
  {
    id: "gigaam-v3-e2e-rnnt",
    where: "onnx-asr",
    languages: "ru, en",
    punctuation: true,
    diskMib: null,
    wer: null,
    werSource: null,
    note: "Та же модель, головная ветка RNNT: точнее CTC, медленнее. Не меряли здесь."
  },
  {
    id: "gigaam-multilingual-large-ctc",
    where: "onnx-asr",
    languages: "ru, en, kz, ky, uz — пять, не семьдесят",
    punctuation: false,
    diskMib: 592,
    wer: "3.0 FLEURS ru / 5.1 Common Voice",
    werSource: "карточка модели",
    note: "Единственный вариант без второго движка, если нужен не только русский. Цена: пунктуации и заглавных нет никогда (словарь 71 токен, ноль верхнего регистра), английская нога слабая (9.4 FLEURS против 3.9 у Whisper large-v3), на Mac идёт по медленной ветке. У нас не запускался."
  },
  {
    id: "gigaam-multilingual-ctc",
    where: "onnx-asr",
    languages: "ru, en, kz, ky, uz",
    punctuation: false,
    diskMib: 225,
    wer: "4.4 FLEURS ru",
    werSource: "карточка модели",
    note: "Вдвое меньше большой версии, хуже примерно на 1.4 пункта. Рычаг, если тесен диск."
  },
  {
    id: "nemo-fastconformer-ru-rnnt",
    where: "onnx-asr",
    languages: "ru",
    punctuation: false,
    diskMib: null,
    wer: null,
    werSource: null,
    note: "Русская альтернатива NeMo. Не меряли здесь."
  },
  {
    id: "t-tech/t-one",
    where: "onnx-asr",
    languages: "ru",
    punctuation: false,
    diskMib: 144,
    wer: "5.32 Common Voice 19",
    werSource: "карточка модели",
    note: "Нативно стриминговый, но в этом рантайме работает без стриминга. Телефонная специализация. Наш замер 26.09: 10.4 % против 3.0 % у GigaAM v3 CTC int8 на том же образце."
  },
  {
    id: "alphacep/vosk-model-ru",
    where: "onnx-asr",
    languages: "ru",
    punctuation: false,
    diskMib: null,
    wer: null,
    werSource: null,
    note: "Лёгкая база для прототипов. Не меряли здесь."
  },
  {
    id: "nemo-parakeet-tdt-0.6b-v3",
    where: "onnx-asr",
    languages: "ru и ещё 24, включая en",
    punctuation: true,
    diskMib: 714,
    wer: "1.93 LibriSpeech test-clean (английский)",
    werSource: "карточка модели",
    note: "Лучший английский из найденного, но русского WER вендор не публикует вовсе. CC-BY-4.0."
  },
  {
    id: "nemo-canary-1b-v2",
    where: "onnx-asr",
    languages: "те же 25",
    punctuation: true,
    diskMib: null,
    wer: null,
    werSource: null,
    note: "Больше Parakeet без опубликованного выигрыша по русскому. CC-BY-4.0."
  }
];

/** The model list the settings form shows under the model field. */
export function describeModels() {
  return [
    "Модели для движка onnx-asr — впишите имя в ortModel, поле принимает любой репозиторий:",
    ...MODELS.map((model) => {
      const parts = [
        model.id,
        `языки: ${model.languages}`,
        model.punctuation ? "пунктуация: да" : "пунктуация: нет",
        model.diskMib === null ? "вес не измерен" : `вес ${model.diskMib} МБ`
      ];
      const wer = model.wer === null ? "WER не измерен" : `WER ${model.wer} (${model.werSource})`;
      return `- ${parts.join(" · ")}. ${wer}. ${model.note}`;
    })
  ].join("\n");
}
