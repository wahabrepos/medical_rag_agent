import { percent } from "@/lib/format";
import type { ProgressEvent } from "@/lib/types";

interface Step {
  iteration: number;
  passages?: number;
  drafted: boolean;
  draft?: string;
  support?: number;
  refining: boolean;
}

export function steps(events: ProgressEvent[]): Step[] {
  const byIteration = new Map<number, Step>();
  const step = (iteration: number) => {
    let s = byIteration.get(iteration);
    if (!s) {
      s = { iteration, drafted: false, refining: false };
      byIteration.set(iteration, s);
    }
    return s;
  };
  for (const e of events) {
    if (e.event === "retrieved") step(e.data.iteration).passages = e.data.passages;
    if (e.event === "generated") {
      const s = step(e.data.iteration);
      s.drafted = true;
      s.draft = e.data.draft_answer;
    }
    if (e.event === "verified") step(e.data.iteration).support = e.data.support;
    if (e.event === "refining") step(e.data.next_iteration - 1).refining = true;
  }
  return [...byIteration.values()].sort((a, b) => a.iteration - b.iteration);
}

export function Progress({ events, busy }: { events: ProgressEvent[]; busy: boolean }) {
  const list = steps(events);
  const rewritten = events.find((e) => e.event === "rewritten");
  return (
    <section className="card progress" aria-label="Progress">
      <h2 className="section-title">
        {busy ? "Searching and checking the literature…" : "How the answer was found"}
      </h2>
      {rewritten?.event === "rewritten" && (
        <p className="searched">
          Also searched in clinical terms: <strong>{rewritten.data.query}</strong>
        </p>
      )}
      <ol aria-live="polite">
        {list.length === 0 && busy && <li className="muted">Starting…</li>}
        {list.map((s) => (
          <li key={s.iteration}>
            <span className="step-title">Attempt {s.iteration}</span>
            <ul>
              {s.passages !== undefined && <li>Retrieved {s.passages} passages</li>}
              {s.drafted && (
                <li>
                  Drafted an answer
                  {s.draft !== undefined && (
                    <span className="unverified-inline"> “{s.draft}” (unverified draft)</span>
                  )}
                </li>
              )}
              {s.support !== undefined && (
                <li>Quick check: {percent(s.support)} of its statements matched the passages</li>
              )}
              {s.refining && <li>Refining the search for the unsupported statements</li>}
            </ul>
          </li>
        ))}
      </ol>
      {!busy && list.length > 0 && (
        <p className="hint">
          The quick checks steer the search. What is shown is decided by the final check below:
          each statement needs a verbatim quote from a source, or must restate your question.
        </p>
      )}
    </section>
  );
}
