"use client";

import { useState, type FormEvent } from "react";

import type { Example } from "@/lib/api";
import type { AnswerFormat, AskRequest } from "@/lib/types";

const FORMATS: { value: AnswerFormat; label: string }[] = [
  { value: "free", label: "Free text" },
  { value: "yes_no", label: "Yes / no" },
  { value: "multiple_choice", label: "Multiple choice" },
];

interface Props {
  examples: Example[];
  busy: boolean;
  onAsk: (request: AskRequest) => void;
  onStop: () => void;
}

export function AskForm({ examples, busy, onAsk, onStop }: Props) {
  const [question, setQuestion] = useState("");
  const [format, setFormat] = useState<AnswerFormat>("free");
  const [includeUnverified, setIncludeUnverified] = useState(false);
  const tooShort = question.trim().length < 3;

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!tooShort && !busy) {
      onAsk({ question: question.trim(), answer_format: format, include_unverified: includeUnverified });
    }
  }

  return (
    <form className="card ask" onSubmit={submit} aria-label="Ask a question">
      {examples.length > 0 && (
        <fieldset className="examples">
          <legend>Example questions (demo mode replays recorded answers)</legend>
          <div className="chips">
            {examples.map((example) => (
              <button
                key={example.id}
                type="button"
                className="chip"
                disabled={busy}
                onClick={() => {
                  setQuestion(example.question);
                  setFormat(example.answerFormat);
                }}
              >
                {example.label}
              </button>
            ))}
          </div>
        </fieldset>
      )}

      <label htmlFor="question" className="label">
        Question
      </label>
      <textarea
        id="question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={6}
        maxLength={4000}
        placeholder="e.g. Is metformin associated with a lower risk of dementia in type 2 diabetes?"
        disabled={busy}
      />
      {format === "multiple_choice" && (
        <p className="hint">
          List the options after the question, one per line, below a line “Answer choices:” (A. …,
          B. …).
        </p>
      )}

      <div className="row">
        <label className="field">
          <span className="label">Answer format</span>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value as AnswerFormat)}
            disabled={busy}
          >
            {FORMATS.map((f) => (
              <option key={f.value} value={f.value}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={includeUnverified}
            onChange={(e) => setIncludeUnverified(e.target.checked)}
            disabled={busy}
          />
          Also show the model’s answer when the literature does not support it (marked unverified)
        </label>
      </div>

      <div className="actions">
        {busy ? (
          <button type="button" className="button secondary" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" className="button" disabled={tooShort}>
            Ask
          </button>
        )}
      </div>
    </form>
  );
}
