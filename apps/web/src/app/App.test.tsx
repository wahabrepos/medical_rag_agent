import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "@/app/App";
import { createMockClient } from "@/lib/api";

function setup() {
  const client = createMockClient(0);
  const feedback = vi.spyOn(client, "feedback");
  render(<App client={client} />);
  return { user: userEvent.setup(), feedback };
}

async function askExample(user: ReturnType<typeof userEvent.setup>, label: RegExp, unverified = false) {
  await user.click(screen.getByRole("button", { name: label }));
  if (unverified) await user.click(screen.getByRole("checkbox"));
  await user.click(screen.getByRole("button", { name: "Ask" }));
  return screen.findByRole("article", { name: "Answer" });
}

describe("App (demo mode)", () => {
  it("always shows the disclaimer", () => {
    setup();
    expect(screen.getByRole("note", { name: "Disclaimer" })).toHaveTextContent(/not medical advice/i);
  });

  it("shows a grounded answer with its quotes linked to the sources", async () => {
    const { user } = setup();
    const card = await askExample(user, /every statement quoted/i);

    expect(within(card).getByRole("heading", { level: 2 })).toHaveTextContent("No");
    const quotes = card.querySelectorAll("blockquote");
    expect(quotes.length).toBeGreaterThan(0);
    const link = within(quotes[0] as HTMLElement).getByRole("link");
    expect(card.querySelector(link.getAttribute("href")!)).not.toBeNull();
    expect(card.querySelector("mark")).not.toBeNull(); // quote highlighted in its passage
    // passages no statement quotes are listed separately, collapsed
    expect(within(card).getByText(/Other retrieved passages \(\d+\)/)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Progress" })).toHaveTextContent(/Attempt 1/);
  });

  it("labels statements that restate the question, which alone do not ground an answer", async () => {
    const { user } = setup();
    const card = await askExample(user, /finding restated from the question/i);
    expect(within(card).getByText("From your question")).toBeInTheDocument();
    expect(within(card).getAllByText(/^Source \[\d+\]$/).length).toBeGreaterThan(0);
    expect(within(card).getByText("Not found in the sources")).toBeInTheDocument();
    expect(within(card).getByRole("heading", { level: 2 })).toHaveTextContent("Insufficient evidence");
  });

  it("withholds an unsupported answer and never leaks it by default", async () => {
    const { user } = setup();
    const card = await askExample(user, /clinical reasoning not found/i);

    expect(within(card).getByRole("heading", { level: 2 })).toHaveTextContent("Insufficient evidence");
    expect(within(card).queryByText(/unverified answer/i)).toBeNull();
    expect(screen.getByRole("region", { name: "Progress" })).not.toHaveTextContent(/unverified draft/);
  });

  it("shows the unverified answer only on request, clearly marked", async () => {
    const { user } = setup();
    const card = await askExample(user, /clinical reasoning not found/i, true);

    expect(within(card).getByText(/model’s unverified answer/i)).toBeInTheDocument();
    expect(within(card).getByText(/Do not rely on it/)).toBeInTheDocument();
  });

  it("marks textbook sources and shows the benchmark answer in demo mode", async () => {
    const { user } = setup();
    const card = await askExample(user, /textbook source/i);
    expect(within(card).getAllByText(/Textbook \(research evaluation only\)/).length).toBeGreaterThan(0);
    expect(within(card).getAllByText(/Harrison's Principles of Internal Medicine/).length).toBeGreaterThan(0);
    // grounded but wrong: the demo says so
    expect(within(card).getByRole("heading", { level: 2 })).toHaveTextContent(/^C\./);
    expect(within(card).getByRole("note")).toHaveTextContent(/benchmark’s answer to this question is B\./);
  });

  it("sends feedback with a comment", async () => {
    const { user, feedback } = setup();
    const card = await askExample(user, /every statement quoted/i);

    await user.click(within(card).getByRole("button", { name: "Not helpful" }));
    await user.type(within(card).getByLabelText(/Comment/), "missing a key study");
    await user.click(within(card).getByRole("button", { name: "Send feedback" }));

    expect(await within(card).findByRole("status")).toHaveTextContent(/recorded/);
    expect(feedback).toHaveBeenCalledWith(expect.any(String), -1, "missing a key study");
  });

  it("explains that demo mode only answers the examples", async () => {
    const { user } = setup();
    await user.type(screen.getByLabelText("Question"), "Does coffee prevent gout?");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/example questions/);
  });
});
