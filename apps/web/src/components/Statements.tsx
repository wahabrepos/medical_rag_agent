import { percent } from "@/lib/format";
import type { StatementEvidence } from "@/lib/types";

function Badge({ statement }: { statement: StatementEvidence }) {
  if (statement.supporting_citation !== null && statement.supported) {
    return <span className="badge ok">Source [{statement.supporting_citation + 1}]</span>;
  }
  if (statement.from_question) return <span className="badge info">From your question</span>;
  return <span className="badge bad">Not found in the sources</span>;
}

export function Statements({ statements }: { statements: StatementEvidence[] }) {
  if (statements.length === 0) {
    return <p className="muted">The model gave no reasoning that could be checked.</p>;
  }
  return (
    <ol className="statements">
      {statements.map((s, i) => {
        const sourced = s.supporting_citation !== null && s.supported;
        return (
          <li key={i} className={sourced ? "sourced" : s.from_question ? "question" : "unsupported"}>
            <div className="statement-head">
              <Badge statement={s} />
              {s.kind === "claim" && <span className="badge neutral">Answer claim</span>}
              {!s.from_question && (
                <span className="score" title="Verifier support for this statement">
                  support {percent(s.support)}
                </span>
              )}
            </div>
            <p className="statement-text">{s.text}</p>
            {sourced && s.quote && (
              <blockquote>
                “{s.quote}”{" "}
                <a href={`#citation-${(s.supporting_citation ?? 0) + 1}`}>
                  [{(s.supporting_citation ?? 0) + 1}]
                </a>
              </blockquote>
            )}
          </li>
        );
      })}
    </ol>
  );
}
