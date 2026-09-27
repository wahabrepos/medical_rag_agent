import { SseParser } from "@/lib/sse";

describe("SseParser", () => {
  it("parses events split across chunks", () => {
    const parser = new SseParser();
    expect(parser.push('event: retrieved\ndata: {"iter')).toEqual([]);
    expect(parser.push('ation": 1}\n\nevent: answer\n')).toEqual([
      { event: "retrieved", data: '{"iteration": 1}' },
    ]);
    expect(parser.push("data: {}\n\n")).toEqual([{ event: "answer", data: "{}" }]);
  });

  it("handles CRLF, comments, multi-line data and a default event name", () => {
    const parser = new SseParser();
    const messages = parser.push(": keep-alive\r\n\r\ndata: a\r\ndata: b\r\n\r\n");
    expect(messages).toEqual([{ event: "message", data: "a\nb" }]);
  });
});
