import fixtures from "@/mocks/fixtures.json";
import { ApiError, createApiClient, mockResponse } from "@/lib/api";
import { INSUFFICIENT_EVIDENCE, type AskResponse, type ProgressEvent } from "@/lib/types";

const ANSWER = (fixtures as unknown as { answer: AskResponse }[])[0]!.answer;

function streamResponse(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { status, headers: { "Content-Type": "text/event-stream" } });
}

afterEach(() => vi.unstubAllGlobals());

describe("createApiClient", () => {
  it("streams progress, returns the answer and sends the key", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      streamResponse([
        'event: retrieved\ndata: {"iteration": 1, "passages": 5}\n\nevent: gen',
        'erated\ndata: {"iteration": 1}\n\n',
        `event: answer\ndata: ${JSON.stringify(ANSWER)}\n\n`,
      ]),
    );
    vi.stubGlobal("fetch", fetchMock);
    const progress: ProgressEvent[] = [];

    const answer = await createApiClient("http://api.test/", () => "k1").ask(
      { question: "Q?", answer_format: "free", include_unverified: false },
      (e) => progress.push(e),
    );

    expect(answer.run_id).toBe(ANSWER.run_id);
    expect(progress.map((e) => e.event)).toEqual(["retrieved", "generated"]);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://api.test/v1/ask/stream");
    expect(init.headers.Authorization).toBe("Bearer k1");
  });

  it("turns error events and HTTP errors into readable messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(streamResponse(['event: error\ndata: {"detail": "LLM budget used up"}\n\n'])),
    );
    const client = createApiClient("http://api.test", () => "");
    const request = { question: "Q?", answer_format: "free" as const, include_unverified: false };
    await expect(client.ask(request, () => {})).rejects.toThrow("LLM budget used up");

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}", { status: 401 })));
    await expect(client.ask(request, () => {})).rejects.toThrow(/API key/);

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network")));
    await expect(client.ask(request, () => {})).rejects.toBeInstanceOf(ApiError);
  });

  it("posts feedback", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"status": "recorded"}', { status: 201 }));
    vi.stubGlobal("fetch", fetchMock);
    await createApiClient("http://api.test", () => "").feedback("r1", -1, "  wrong  ");
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://api.test/v1/runs/r1/feedback");
    expect(JSON.parse(init.body)).toEqual({ rating: -1, comment: "wrong" });
  });
});

describe("mockResponse", () => {
  const withheld = (fixtures as unknown as { answer: AskResponse }[]).find(
    (f) => f.answer.answer === INSUFFICIENT_EVIDENCE,
  )!.answer;

  it("withholds the model's answer unless it was requested, like the API", () => {
    const hidden = mockResponse(withheld, { question: "", answer_format: "free", include_unverified: false });
    expect(hidden.model_answer).toBeNull();
    expect(hidden.note).not.toMatch(/Unverified/);

    const shown = mockResponse(withheld, { question: "", answer_format: "free", include_unverified: true });
    expect(shown.model_answer).not.toBeNull();
  });
});
