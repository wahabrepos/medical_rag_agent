import { render, screen } from "@testing-library/react";

import { Citations } from "@/components/Citations";
import { Progress } from "@/components/Progress";
import type { Citation, StatementEvidence } from "@/lib/types";

describe("Progress", () => {
  it("shows the clinical search query", () => {
    render(
      <Progress
        busy
        events={[
          { event: "rewritten", data: { query: "polyuria polydipsia weight loss" } },
          { event: "retrieved", data: { iteration: 1, passages: 5 } },
        ]}
      />,
    );
    expect(screen.getByText(/Also searched in clinical terms/)).toHaveTextContent(
      "polyuria polydipsia weight loss",
    );
  });
});

describe("Citations", () => {
  it("labels and links MedlinePlus topics, with NLM's attribution", () => {
    const citation: Citation = {
      source: "medlineplus",
      pmid: null,
      title: "Diabetes",
      url: "https://medlineplus.gov/diabetes.html",
      passage: "Diabetes: What are the symptoms of diabetes? Feeling very thirsty.",
    };
    const statement: StatementEvidence = {
      kind: "rationale",
      text: "Thirst is a symptom of diabetes",
      support: 0.9,
      supported: true,
      contradicted: false,
      supporting_pmid: null,
      contradicting_pmid: null,
      supporting_citation: 0,
      quote: "What are the symptoms of diabetes? Feeling very thirsty",
      from_question: false,
    };
    render(<Citations citations={[citation]} statements={[statement]} />);

    expect(screen.getByRole("link", { name: "Diabetes" })).toHaveAttribute(
      "href",
      "https://medlineplus.gov/diabetes.html",
    );
    expect(screen.getByText("MedlinePlus (NLM)")).toBeInTheDocument();
    expect(screen.getByText(/Courtesy of MedlinePlus from the National Library of Medicine/)).toBeInTheDocument();
  });
});
