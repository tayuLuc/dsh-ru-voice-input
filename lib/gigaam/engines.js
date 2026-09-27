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
