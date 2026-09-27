import type { AnswerFormat } from "@/lib/types";

export interface Option {
  letter: string;
  text: string;
}

const CHOICES_MARKER = /\n\s*Answer choices:\s*\n/i;
const OPTION_LINE = /^\s*([A-E])[.)]\s+(.+?)\s*$/;

/** Splits "stem\n\nAnswer choices:\nA. ...\nB. ..." (the format the API expects). */
export function parseQuestion(question: string): { stem: string; options: Option[] } {
  const match = CHOICES_MARKER.exec(question);
  if (!match) return { stem: question.trim(), options: [] };
  const options: Option[] = [];
  for (const line of question.slice(match.index + match[0].length).split("\n")) {
    const option = OPTION_LINE.exec(line);
    if (option?.[1] && option[2]) options.push({ letter: option[1], text: option[2] });
  }
  return { stem: question.slice(0, match.index).trim(), options };
}

/** The answer as users read it: "B. Borderline personality disorder", "Yes", ... */
export function answerLabel(answer: string, format: AnswerFormat, options: Option[]): string {
  const text = answer.trim();
  if (format === "yes_no") return text.charAt(0).toUpperCase() + text.slice(1);
  if (format === "multiple_choice") {
    const letter = /^([A-E])\b/i.exec(text)?.[1]?.toUpperCase();
    const option = options.find((o) => o.letter === letter);
    if (option) return `${option.letter}. ${option.text}`;
  }
  return text;
}

export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** Splits a passage around a quote (case- and whitespace-insensitive) for highlighting. */
export function splitOnQuote(
  passage: string,
  quote: string | null,
): [before: string, match: string, after: string] | null {
  if (!quote) return null;
  const words = quote
    .replace(/\s*(\.\.\.|…)\s*/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (words.length < 3) return null;
  const pattern = new RegExp(words.join("[\\s\\S]{0,3}?\\s*"), "i");
  const found = pattern.exec(passage);
  if (!found) return null;
  return [
    passage.slice(0, found.index),
    found[0],
    passage.slice(found.index + found[0].length),
  ];
}
