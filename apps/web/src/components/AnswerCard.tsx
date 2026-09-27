import { Citations } from "@/components/Citations";
import { Feedback } from "@/components/Feedback";
import { Statements } from "@/components/Statements";
import type { AnswerClient } from "@/lib/api";
import { answerLabel, parseQuestion } from "@/lib/format";
import { INSUFFICIENT_EVIDENCE, type AskResponse } from "@/lib/types";

interface Props {
  response: AskResponse;
  client: AnswerClient;
  /** Demo mode: the benchmark's correct answer to a recorded example. */
  benchmarkAnswer?: string;
}

export function AnswerCard({ response, client, benchmarkAnswer }: Props) {
  const { options } = parseQuestion(response.question);
  const withheld = response.answer === INSUFFICIENT_EVIDENCE;
  const note = response.note?.replace(/ Unverified model answer .*$/, "");

  return (
    <article className="card answer" aria-label="Answer">
      {withheld ? (
        <section className="verdict withheld">
          <h2>Insufficient evidence</h2>
          <p>{note ?? "The retrieved studies do not support an answer."}</p>
          {response.model_answer && (
            <details className="unverified">
              <summary>Show the model’s unverified answer</summary>
              <p>
                <strong>
                  {answerLabel(response.model_answer, response.answer_format, options)}
                </strong>{" "}
                — not backed by the retrieved studies. Do not rely on it.
              </p>
            </details>
          )}
        </section>
      ) : (
        <section className="verdict shown">
          <h2 className="answer-text">
            {answerLabel(response.answer, response.answer_format, options)}
          </h2>
          <p>
            Every statement below is quoted from a retrieved source and checked, or restates your
            question.
          </p>
        </section>
      )}

      <section>
        <h3 className="section-title">Reasoning, checked against the sources</h3>
        {(response.evidence.statements.length > 0 || !response.evidence.hidden_statements) && (
          <Statements statements={response.evidence.statements} />
        )}
        {response.evidence.hidden_statements > 0 && (
          <p className="hint">
            {response.evidence.hidden_statements} unchecked{" "}
            {response.evidence.hidden_statements === 1 ? "statement is" : "statements are"} hidden
            because they would reveal the withheld answer. Tick “Also show the model’s answer…” to
            see them, marked as unverified.
          </p>
        )}
      </section>

      <section>
        <h3 className="section-title">Sources</h3>
        <Citations citations={response.citations} statements={response.evidence.statements} />
      </section>

      {benchmarkAnswer && (
        <p className="benchmark" role="note">
          Demo: the benchmark’s answer to this question is{" "}
          <strong>{answerLabel(benchmarkAnswer, response.answer_format, options)}</strong>. Grounded
          statements show where each claim comes from; they do not guarantee the conclusion.
        </p>
      )}
      <p className="meta">
        {response.iterations} {response.iterations === 1 ? "attempt" : "attempts"} ·{" "}
        {response.latency_seconds.toFixed(1)} s · model {response.model}
      </p>
      <p className="meta">{response.disclaimer}</p>
      <Feedback key={response.run_id} client={client} runId={response.run_id} />
    </article>
  );
}
