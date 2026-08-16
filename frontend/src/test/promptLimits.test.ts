import { describe, expect, it } from "vitest";
import {
  limitPromptCharacters,
  MINIMAX_H3_PROMPT_MAX_CHARACTERS,
  promptCharacterCount,
} from "../domain/promptLimits";

describe("MiniMax H3 prompt character limit", () => {
  it("counts Chinese characters independently of their UTF-8 byte size", () => {
    expect(promptCharacterCount("中文。A")).toBe(4);
    expect(promptCharacterCount("中".repeat(MINIMAX_H3_PROMPT_MAX_CHARACTERS))).toBe(7_000);
  });

  it("counts supplementary Unicode characters as one code point and truncates safely", () => {
    const prompt = `😀${"中".repeat(MINIMAX_H3_PROMPT_MAX_CHARACTERS)}`;
    expect(prompt.length).toBe(MINIMAX_H3_PROMPT_MAX_CHARACTERS + 2);
    expect(promptCharacterCount(prompt)).toBe(MINIMAX_H3_PROMPT_MAX_CHARACTERS + 1);
    expect(limitPromptCharacters(prompt)).toBe(`😀${"中".repeat(6_999)}`);
  });
});
