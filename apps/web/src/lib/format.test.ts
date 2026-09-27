import { answerLabel, parseQuestion, percent, splitOnQuote } from "@/lib/format";

const MCQ = "Which drug?\n\nAnswer choices:\nA. Aspirin\nB. Warfarin\nC. Heparin";

describe("parseQuestion", () => {
  it("splits the stem from the options", () => {
    expect(parseQuestion(MCQ)).toEqual({
      stem: "Which drug?",
      options: [
        { letter: "A", text: "Aspirin" },
        { letter: "B", text: "Warfarin" },
        { letter: "C", text: "Heparin" },
      ],
    });
  });

  it("returns no options for other questions", () => {
    expect(parseQuestion("  Does X help?  ")).toEqual({ stem: "Does X help?", options: [] });
  });
});

describe("answerLabel", () => {
  const { options } = parseQuestion(MCQ);
  it("names the option of a multiple-choice answer", () => {
    expect(answerLabel("B", "multiple_choice", options)).toBe("B. Warfarin");
    expect(answerLabel("b) something", "multiple_choice", options)).toBe("B. Warfarin");
  });
  it("capitalises yes/no and leaves free text", () => {
    expect(answerLabel("no", "yes_no", [])).toBe("No");
    expect(answerLabel("insufficient evidence", "free", [])).toBe("insufficient evidence");
  });
});

describe("splitOnQuote", () => {
  const passage = "Background text. Private referral rates were lower in the most deprived areas. More.";
  it("finds a quote ignoring case and spacing", () => {
    expect(splitOnQuote(passage, "private referral  rates were LOWER in the most deprived areas")).toEqual([
      "Background text. ",
      "Private referral rates were lower in the most deprived areas",
      ". More.",
    ]);
  });
  it("returns null when the quote is absent or too short", () => {
    expect(splitOnQuote(passage, "rates were higher everywhere")).toBeNull();
    expect(splitOnQuote(passage, "lower")).toBeNull();
    expect(splitOnQuote(passage, null)).toBeNull();
  });
});

it("formats percentages", () => {
  expect(percent(0.9858)).toBe("99%");
});
