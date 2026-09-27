"use client";

import { useState } from "react";

import type { AnswerClient } from "@/lib/api";

type State = "idle" | "sending" | "sent" | "error";

export function Feedback({ client, runId }: { client: AnswerClient; runId: string }) {
  const [rating, setRating] = useState<1 | -1 | null>(null);
  const [comment, setComment] = useState("");
  const [state, setState] = useState<State>("idle");
  const [error, setError] = useState("");

  if (state === "sent") {
    return <p className="feedback-done" role="status">Thank you — your feedback was recorded.</p>;
  }

  async function send() {
    if (rating === null) return;
    setState("sending");
    try {
      await client.feedback(runId, rating, comment);
      setState("sent");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not send feedback.");
      setState("error");
    }
  }

  return (
    <div className="feedback">
      <span className="label" id="feedback-label">
        Was this helpful?
      </span>
      <div className="thumbs" role="group" aria-labelledby="feedback-label">
        <button
          type="button"
          className="thumb"
          aria-pressed={rating === 1}
          aria-label="Helpful"
          onClick={() => setRating(1)}
        >
          👍
        </button>
        <button
          type="button"
          className="thumb"
          aria-pressed={rating === -1}
          aria-label="Not helpful"
          onClick={() => setRating(-1)}
        >
          👎
        </button>
      </div>
      {rating !== null && (
        <>
          <label htmlFor="feedback-comment" className="label">
            Comment (optional)
          </label>
          <textarea
            id="feedback-comment"
            rows={2}
            maxLength={2000}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
          <button type="button" className="button" disabled={state === "sending"} onClick={send}>
            {state === "sending" ? "Sending…" : "Send feedback"}
          </button>
        </>
      )}
      {state === "error" && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
