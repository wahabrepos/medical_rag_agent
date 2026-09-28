import { splitOnQuote } from "@/lib/format";
import type { Citation, StatementEvidence } from "@/lib/types";

function Passage({ text, quotes }: { text: string; quotes: string[] }) {
  for (const quote of quotes) {
    const parts = splitOnQuote(text, quote);
    if (parts) {
      return (
        <p className="passage">
          {parts[0]}
          <mark>{parts[1]}</mark>
          {parts[2]}
        </p>
      );
    }
  }
  return <p className="passage">{text}</p>;
}

const SOURCE_LABELS: Record<string, string> = {
  textbook: "Textbook (research evaluation only)",
  medlineplus: "MedlinePlus (NLM)",
};

function Item({ citation, index, quotes }: { citation: Citation; index: number; quotes: string[] }) {
  return (
    <li id={`citation-${index + 1}`}>
      <div className="citation-head">
        <span className="cite-number">[{index + 1}]</span>
        {citation.url ? (
          <a href={citation.url} target="_blank" rel="noopener noreferrer">
            {citation.title}
          </a>
        ) : (
          <span>{citation.title}</span>
        )}
        {citation.source === "pubmed" && citation.pmid !== null && (
          <span className="muted">PMID {citation.pmid}</span>
        )}
        {citation.source !== "pubmed" && (
          <span className="badge neutral" title="Not a PubMed study">
            {SOURCE_LABELS[citation.source] ?? citation.source}
          </span>
        )}
        {quotes.length > 0 && <span className="badge ok">quoted</span>}
      </div>
      <details open={quotes.length > 0}>
        <summary>Passage</summary>
        <Passage text={citation.passage} quotes={quotes} />
      </details>
    </li>
  );
}

/** Sources quoted by a statement first; the other retrieved passages collapsed below. */
export function Citations({
  citations,
  statements,
}: {
  citations: Citation[];
  statements: StatementEvidence[];
}) {
  if (citations.length === 0) return <p className="muted">No sources were retrieved.</p>;
  const quotesOf = (i: number) =>
    statements.filter((s) => s.supporting_citation === i && s.quote).map((s) => s.quote as string);
  const cited = citations.map((c, i) => ({ c, i })).filter(({ i }) => quotesOf(i).length > 0);
  const others = citations.map((c, i) => ({ c, i })).filter(({ i }) => quotesOf(i).length === 0);
  return (
    <>
      {cited.length > 0 ? (
        <ol className="citations">
          {cited.map(({ c, i }) => (
            <Item key={i} citation={c} index={i} quotes={quotesOf(i)} />
          ))}
        </ol>
      ) : (
        <p className="muted">No retrieved passage supports the reasoning.</p>
      )}
      {citations.some((c) => c.source === "medlineplus") && (
        <p className="hint">Courtesy of MedlinePlus from the National Library of Medicine.</p>
      )}
      {others.length > 0 && (
        <details className="other-sources">
          <summary>
            Other retrieved passages ({others.length}), not quoted by any statement
          </summary>
          <ol className="citations">
            {others.map(({ c, i }) => (
              <Item key={i} citation={c} index={i} quotes={[]} />
            ))}
          </ol>
        </details>
      )}
    </>
  );
}
