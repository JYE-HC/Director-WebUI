/** MiniMax H3 V2 API limit for each text prompt, measured in Unicode characters. */
export const MINIMAX_H3_PROMPT_MAX_CHARACTERS = 7_000;

/** JavaScript string.length counts UTF-16 code units; iteration counts Unicode code points. */
export function promptCharacterCount(value: string): number {
  return Array.from(value).length;
}

export function limitPromptCharacters(value: string): string {
  if (promptCharacterCount(value) <= MINIMAX_H3_PROMPT_MAX_CHARACTERS) return value;
  return Array.from(value).slice(0, MINIMAX_H3_PROMPT_MAX_CHARACTERS).join("");
}
