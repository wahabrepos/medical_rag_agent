// Mirrors apps/api/src/medrag_api/schemas.py (the API's JSON).

export type AnswerFormat = "free" | "yes_no" | "multiple_choice";

export type EvidenceStatus =
  | "supported"
  | "partially_supported"
  | "not_supported"
  | "contradicted"
  | "no_evidence";

/** What the evidence gate answers when the literature does not ground an answer. */
export const INSUFFICIENT_EVIDENCE = "insufficient evidence";

export interface AskRequest {
  question: string;
  answer_format: AnswerFormat;
  include_unverified: boolean;
}

export interface Citation {
  source: string; // "pubmed" or a knowledge source such as "textbook"
  pmid: number | null;
  title: string;
  url: string | null;
  passage: string;
}

export interface StatementEvidence {
  kind: "claim" | "rationale";
  text: string;
  support: number;
  supported: boolean;
  contradicted: boolean;
  supporting_pmid: number | null;
  contradicting_pmid: number | null;
  supporting_citation: number | null;
  quote: string | null;
  from_question: boolean;
}

export interface Evidence {
  status: EvidenceStatus;
  supported_fraction: number;
  message: string;
  statements: StatementEvidence[];
  /** Ungrounded statements left out of a withheld answer (they would reveal it). */
  hidden_statements: number;
}

export interface AskResponse {
  run_id: string;
  question: string;
  answer_format: AnswerFormat;
  answer: string;
  model_answer: string | null;
  note: string | null;
  evidence: Evidence;
  citations: Citation[];
  iterations: number;
  stop_reason: string;
  model: string;
  latency_seconds: number;
  cost: number;
  disclaimer: string;
}

/** Progress events of POST /v1/ask/stream, before the final "answer" event. */
export type ProgressEvent =
  | { event: "rewritten"; data: { query: string } }
  | { event: "retrieved"; data: { iteration: number; passages: number } }
  | { event: "generated"; data: { iteration: number; draft_answer?: string } }
  | { event: "verified"; data: { iteration: number; support: number; decision?: string | null } }
  | { event: "refining"; data: { next_iteration: number } };

export type StreamEvent =
  | ProgressEvent
  | { event: "answer"; data: AskResponse }
  | { event: "error"; data: { detail: string } };
