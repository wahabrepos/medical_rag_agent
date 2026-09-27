"use client";

import { useMemo, useRef, useState } from "react";

import { AnswerCard } from "@/components/AnswerCard";
import { AskForm } from "@/components/AskForm";
import { Disclaimer } from "@/components/Disclaimer";
import { Progress } from "@/components/Progress";
import { Settings } from "@/components/Settings";
import { ApiError, type AnswerClient, createClient } from "@/lib/api";
import { load, save } from "@/lib/storage";
import type { AskRequest, AskResponse, ProgressEvent } from "@/lib/types";

const API_KEY = "medrag.apiKey";

type Phase =
  | { kind: "idle" }
  | { kind: "running"; events: ProgressEvent[] }
  | { kind: "done"; events: ProgressEvent[]; response: AskResponse }
  | { kind: "error"; events: ProgressEvent[]; message: string };

export function App({ client: injected }: { client?: AnswerClient }) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  // The settings panel (the only place showing the key) is closed on first render,
  // so reading storage here cannot cause a hydration mismatch.
  const [apiKey, setApiKey] = useState(() => (typeof window === "undefined" ? "" : load(API_KEY)));
  // The key is read from storage when a request is sent.
  const client = useMemo(
    () => injected ?? createClient(apiUrl, () => load(API_KEY)),
    [injected, apiUrl],
  );
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const abort = useRef<AbortController | null>(null);

  function updateKey(key: string) {
    setApiKey(key);
    save(API_KEY, key);
  }

  async function ask(request: AskRequest) {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    const events: ProgressEvent[] = [];
    setPhase({ kind: "running", events: [] });
    try {
      const response = await client.ask(
        request,
        (event) => {
          events.push(event);
          setPhase({ kind: "running", events: [...events] });
        },
        controller.signal,
      );
      setPhase({ kind: "done", events, response });
    } catch (error) {
      if (controller.signal.aborted) {
        setPhase({ kind: "idle" });
        return;
      }
      const message =
        error instanceof ApiError ? error.message : "Something went wrong while answering.";
      setPhase({ kind: "error", events, message });
    }
  }

  const busy = phase.kind === "running";
  return (
    <>
      <header className="topbar">
        <div>
          <h1>MedRAG Agent</h1>
          <p className="tagline">
            Answers from the medical literature, shown only when the retrieved studies support them
          </p>
        </div>
        <Settings apiUrl={apiUrl} mock={client.mock} apiKey={apiKey} onApiKey={updateKey} />
      </header>
      <Disclaimer />
      <main>
        <AskForm
          examples={client.examples}
          busy={busy}
          onAsk={ask}
          onStop={() => abort.current?.abort()}
        />
        {phase.kind !== "idle" && (phase.events.length > 0 || busy) && (
          <Progress events={phase.events} busy={busy} />
        )}
        {phase.kind === "error" && (
          <p className="card error" role="alert">
            {phase.message}
          </p>
        )}
        {phase.kind === "done" && (
          <AnswerCard
            response={phase.response}
            client={client}
            benchmarkAnswer={
              client.examples.find((e) => e.question === phase.response.question)?.benchmarkAnswer
            }
          />
        )}
      </main>
    </>
  );
}
