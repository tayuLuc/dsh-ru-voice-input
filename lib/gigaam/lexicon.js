/**
 * Lexicon correction for GigaAM transcripts.
 *
 * The engines here are small Russian character models, and they get domain
 * vocabulary wrong: `Sisyphus` comes back as `сизифус`, `авто-брайн` as
 * `автобрайн`. None of the runtimes supports a hotword bias, so correction
 * happens on the text after recognition.
 *
 * The rule that keeps this from corrupting good transcripts: a term is only
 * written where it is *not already present* and a window of the transcript is
 * *similar enough*. Above `threshold` we replace; below it we leave the text
 * alone, because a wrong replacement is worse than a wrong word the user can
 * read and fix.
 */

/** Fold the differences that are spelling, not meaning. */
export function normalize(text) {
  return text
    .toLowerCase()
    .replace(/ё/g, "е")
    .replace(/[^\p{L}\p{N}\s-]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Drop separators, so a lost hyphen or space does not read as a different word. */
function compact(text) {
  return text.replace(/[\s-]+/g, "");
}

/**
 * Cyrillic to Latin, by the usual letter-to-sound choices.
 *
 * A Russian recogniser cannot know whether a speaker said «Сизифус» or
 * «Sisyphus», so it writes Latin names in Cyrillic. Folding both sides through
 * the same map makes those two spellings compare as what they are.
 */
const TRANSLIT = new Map(Object.entries({
  а: "a", б: "b", в: "v", г: "g", д: "d", е: "e", ё: "e", ж: "zh", з: "z",
  и: "i", й: "y", к: "k", л: "l", м: "m", н: "n", о: "o", п: "p", р: "r",
  с: "s", т: "t", у: "y", ф: "f", х: "kh", ц: "ts", ч: "ch", ш: "sh",
  щ: "shch", ъ: "", ы: "y", ь: "", э: "e", ю: "yu", я: "ya"
}));

function translit(text) {
  return [...text].map((character) => TRANSLIT.get(character) ?? character).join("");
}

/** The forms one spelling may plausibly have been recognised as. */
function variants(text) {
  const folded = normalize(text);
  return {
    plain: folded,
    tight: compact(folded),
    latin: translit(folded),
    latinTight: compact(translit(folded))
  };
}

/**
 * Similarity of two strings in 0..1.
 *
 * The best score across the plausible spellings wins, so a term still matches
 * when the recogniser dropped a hyphen, wrote a Latin name in Cyrillic, or did
 * both at once.
 */
export function similarity(a, b) {
  const left = variants(a);
  const right = variants(b);
  return Math.max(
    rawSimilarity(left.plain, right.plain),
    rawSimilarity(left.tight, right.tight),
    rawSimilarity(left.latin, right.latin),
    rawSimilarity(left.latinTight, right.latinTight)
  );
}

/** Longest common substring ratio: the raw measure underneath. */
function rawSimilarity(a, b) {
  if (a === b) return 1;
  const longest = Math.max(a.length, b.length);
  if (longest === 0) return 1;
  return longestCommonSubstring(a, b) / longest;
}

function longestCommonSubstring(a, b) {
  let best = 0;
  let previous = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i += 1) {
    const current = new Array(b.length + 1).fill(0);
    for (let j = 1; j <= b.length; j += 1) {
      if (a[i - 1] === b[j - 1]) {
        current[j] = previous[j - 1] + 1;
        if (current[j] > best) best = current[j];
      }
    }
    previous = current;
  }
  return best;
}

/**
 * Apply the lexicon to one transcript.
 *
 * @param {string} text - recognised text, returned as-is when no term applies
 * @param {string[]} terms - canonical terms, in the spelling the user wants
 * @param {{ threshold?: number }} [options] - minimum similarity to accept
 * @returns {{ text: string, applied: { from: string, to: string }[] }}
 */
export function correct(text, terms, options = {}) {
  const threshold = options.threshold ?? 0.8;
  const applied = [];
  if (typeof text !== "string" || text.length === 0) return { text, applied };
  const canonical = (terms ?? []).filter((term) => typeof term === "string" && term.trim().length > 1);
  if (canonical.length === 0) return { text, applied };

  // Anything already spelled right stays: a term that is present must never
  // be "corrected" into itself by a neighbouring lookalike.
  const haystack = normalize(text);
  const present = new Set(
    canonical.filter((term) => {
      const needle = normalize(term);
      return needle.length > 0 && haystack.includes(needle);
    })
  );

  for (const term of canonical) {
    const needle = normalize(term);
    if (needle.length === 0) continue;
    if (present.has(term)) continue;
    const words = term.trim().split(/\s+/);
    const width = words.length;
    const tokens = haystack.split(" ");
    for (let start = 0; start + width <= tokens.length; start += 1) {
      const window = tokens.slice(start, start + width).join(" ");
      if (window === needle) continue;
      if (similarity(window, needle) < threshold) continue;
      // Replace the original slice, matched case-insensitively, once.
      const pattern = new RegExp(tokens.slice(start, start + width).map(escapeRegExp).join("\\s+"), "iu");
      if (!pattern.test(text)) continue;
      text = text.replace(pattern, term.trim());
      applied.push({ from: window, to: term.trim() });
      present.add(term);
      break;
    }
  }
  return { text, applied };
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
