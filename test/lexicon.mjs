/**
 * Behavioural tests for lexicon correction.
 *
 * The failure this guards against is not "no correction" but "wrong
 * correction": a dictionary that rewrites text the user already got right is
 * worse than no dictionary at all. So most of these checks are about refusing
 * to act.
 *
 * Run with `node test/lexicon.mjs`.
 */
import assert from "node:assert/strict";
import { correct, normalize, similarity } from "../lib/gigaam/lexicon.js";

let failures = 0;
const check = (name, run) => {
  try {
    run();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures += 1;
    console.log(`FAIL ${name}: ${error.message}`);
  }
};

const TERMS = ["Sisyphus", "авто-брайн", "dsh", "GigaAM", "obsidian"];

check("normalisation folds ё, case and punctuation", () => {
  assert.equal(normalize("Ёжик, ДШ — тест!"), "ежик дш тест");
  assert.equal(normalize("  два   пробела  "), "два пробела");
});

check("similarity is high for near misses and low for unrelated words", () => {
  assert.ok(similarity("автобрайн", "авто-брайн") > 0.9, "hyphen loss is a near miss");
  assert.ok(similarity("кошка", "дорога") < 0.4, "unrelated words stay apart");
});

check("a Latin name written in Cyrillic folds to the same thing", () => {
  // Where the letter mapping is clean, the two spellings are identical.
  assert.equal(similarity("GigaAM", "гигаам"), 1);
  assert.equal(similarity("dsh", "дш"), 1);
});

check("heavy transliteration is left alone rather than guessed at", () => {
  // A recogniser cannot know whether the speaker said «Сизифус» or «Sisyphus».
  // No string measure recovers that reliably, and a wrong rewrite is worse
  // than a wrong word, so the corrector deliberately does not act here.
  const result = correct("Сегодня я говорил с Сизифусом про задачу", TERMS);
  assert.equal(result.text, "Сегодня я говорил с Сизифусом про задачу");
  assert.equal(result.applied.length, 0);
});

check("a transliterated name that folds cleanly is restored", () => {
  const result = correct("сегодня ставил дш на хост", TERMS);
  assert.equal(result.text, "сегодня ставил dsh на хост");
  assert.equal(result.applied.length, 1);
});

check("a term already spelled correctly is left alone", () => {
  const text = "Работаю в Sisyphus уже давно";
  const result = correct(text, TERMS);
  assert.equal(result.text, text);
  assert.equal(result.applied.length, 0);
});

check("a hyphen lost by the recogniser is restored", () => {
  const result = correct("лежал в автобрайн уже неделю", TERMS);
  assert.equal(result.text, "лежал в авто-брайн уже неделю");
});

check("unrelated words are never rewritten", () => {
  const text = "Сегодня мы сравниваем локальные движки распознавания речи";
  const result = correct(text, TERMS);
  assert.equal(result.text, text);
  assert.equal(result.applied.length, 0);
});

check("a threshold of 1.0 makes the corrector inert", () => {
  const text = "Сегодня я говорил с Сизифусом";
  const result = correct(text, TERMS, { threshold: 1 });
  assert.equal(result.text, text);
});

check("an empty term list is a no-op", () => {
  const text = "любой текст";
  assert.equal(correct(text, []).text, text);
  assert.equal(correct(text, undefined).text, text);
});

check("multi-word terms match as a phrase, not as single words", () => {
  const result = correct("открыл авто мозг и смотрел", ["авто-брайн"]);
  // The two words stay separate: a phrase must not be stitched from pieces
  // that merely look close on their own.
  assert.equal(result.text, "открыл авто мозг и смотрел");
});

check("a term longer than the transcript cannot match", () => {
  const result = correct("да", ["совершенно другой термин"]);
  assert.equal(result.text, "да");
});

if (failures > 0) {
  console.log(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("all checks passed");
