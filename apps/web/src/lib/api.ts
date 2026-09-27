import fixtures from "@/mocks/fixtures.json";
import { SseParser } from "@/lib/sse";
import {
  INSUFFICIENT_EVIDENCE,
  type AskRequest,
  type AskResponse,
  type ProgressEvent,
  type StreamEvent,
} from "@/lib/types";

export class ApiError extends Error {}

export interface Example {
  id: string;
  label: string;
  question: string;
  answerFormat: AskRequest["answer_format"];
  /** The benchmark's correct answer, shown in demo mode for comparison. */
  benchmarkAnswer: string;
}

export interface AnswerClient {
  /** Replays recorded answers instead of calling the API. */
  readonly mock: boolean;
  /** Questions the mock can answer (empty for the real API). */
  readonly examples: Example[];
  ask(
    request: AskRequest,
    onProgress: (event: ProgressEvent) => void,
    signal?: AbortSignal,
  ): Promise<AskResponse>;
  feedback(runId: string, rating: 1 | -1, comment: string): Promise<void>;
}

const STATUS_MESSAGES: Record<number, string> = {
  401: "The API rejected the key. Set a valid API key in Settings.",
  429: "Too many questions in a minute. Wait a moment and try again.",
  503: "The service is temporarily unavailable (model provider, verifier or budget).",
};

async function failure(response: Response): Promise<ApiError> {
  let detail = "";
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // no JSON body
  }
  const known = STATUS_MESSAGES[response.status];
  return new ApiError(known ?? `Request failed (${response.status})${detail ? `: ${detail}` : ""}`);
}

export function createApiClient(baseUrl: string, apiKey: () => string): AnswerClient {
  const url = (path: string) => `${baseUrl.replace(/\/+$/, "")}${path}`;
  const headers = (): Record<string, string> => {
    const key = apiKey().trim();
    return {
      "Content-Type": "application/json",
      ...(key ? { Authorization: `Bearer ${key}` } : {}),
    };
  };

  return {
    mock: false,
    examples: [],
    async ask(request, onProgress, signal) {
      let response: Response;
      try {
        response = await fetch(url("/v1/ask/stream"), {
          method: "POST",
          headers: headers(),
          body: JSON.stringify(request),
          signal,
        });
      } catch (error) {
        if (signal?.aborted) throw error;
        throw new ApiError(`Cannot reach the API at ${baseUrl}.`);
      }
      if (!response.ok || !response.body) throw await failure(response);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const parser = new SseParser();
      for (;;) {
        const { done, value } = await reader.read();
        const messages = parser.push(done ? decoder.decode() : decoder.decode(value, { stream: true }));
        for (const message of messages) {
          const event = { event: message.event, data: JSON.parse(message.data) } as StreamEvent;
          if (event.event === "answer") return event.data;
          if (event.event === "error") throw new ApiError(event.data.detail);
          onProgress(event);
        }
        if (done) break;
      }
      throw new ApiError("The answer stream ended without an answer.");
    },
    async feedback(runId, rating, comment) {
      const response = await fetch(url(`/v1/runs/${runId}/feedback`), {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({ rating, comment: comment.trim() || null }),
      });
      if (!response.ok) throw await failure(response);
    },
  };
}

interface Fixture {
  id: string;
  label: string;
  gold_answer: string;
  request: { question: string; answer_format: AskRequest["answer_format"] };
  events: ProgressEvent[];
  answer: AskResponse;
}

const UNVERIFIED_NOTE = / Unverified model answer \(not backed by these studies\): ".*"\.$/;

/** What the API returns for a recorded answer, given the request's options. */
export function mockResponse(answer: AskResponse, request: AskRequest): AskResponse {
  if (answer.answer !== INSUFFICIENT_EVIDENCE || request.include_unverified) return answer;
  return {
    ...answer,
    model_answer: null,
    note: answer.note ? answer.note.replace(UNVERIFIED_NOTE, "") : answer.note,
  };
}

export function createMockClient(stepMs = 450): AnswerClient {
  const recorded = fixtures as unknown as Fixture[];
  const sleep = (ms: number, signal?: AbortSignal) =>
    new Promise<void>((resolve, reject) => {
      const timer = setTimeout(resolve, ms);
      signal?.addEventListener("abort", () => {
        clearTimeout(timer);
        reject(new DOMException("aborted", "AbortError"));
      });
    });

  return {
    mock: true,
    examples: recorded.map((f) => ({
      id: f.id,
      label: f.label,
      question: f.request.question,
      answerFormat: f.request.answer_format,
      benchmarkAnswer: f.gold_answer,
    })),
    async ask(request, onProgress, signal) {
      const fixture = recorded.find((f) => f.request.question.trim() === request.question.trim());
      if (!fixture) {
        throw new ApiError(
          "Demo mode replays recorded answers: pick one of the example questions, or connect the UI to the API.",
        );
      }
      for (const event of fixture.events) {
        await sleep(stepMs, signal);
        onProgress(
          event.event === "generated" && !request.include_unverified
            ? { event: "generated", data: { iteration: event.data.iteration } }
            : event,
        );
      }
      await sleep(stepMs, signal);
      return mockResponse(fixture.answer, request);
    },
    async feedback() {
      await sleep(stepMs / 2);
    },
  };
}

export function createClient(baseUrl: string | undefined, apiKey: () => string): AnswerClient {
  return !baseUrl || baseUrl === "mock" ? createMockClient() : createApiClient(baseUrl, apiKey);
}
