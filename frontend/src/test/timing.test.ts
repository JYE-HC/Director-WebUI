import { createInitialDrafts } from "../domain/modes";
import {
  alignH3FrameCount,
  alignH3Frames,
  effectiveDraftTiming,
  effectiveShotTiming,
  roundPositiveHalfEven,
} from "../domain/timing";

describe("Python 一致的 MiniMax H3 帧换算", () => {
  it.each([
    [0.5, 0],
    [1.5, 2],
    [2.5, 2],
    [3.5, 4],
    [22.5, 22],
    [23.5, 24],
  ])("half-even round(%s) = %s", (value, expected) => {
    expect(roundPositiveHalfEven(value)).toBe(expected);
  });

  it("拒绝负数和非有限值，而不是静默使用 JavaScript round", () => {
    expect(() => roundPositiveHalfEven(-0.5)).toThrow(RangeError);
    expect(() => roundPositiveHalfEven(Number.NaN)).toThrow(RangeError);
    expect(() => roundPositiveHalfEven(Number.POSITIVE_INFINITY)).toThrow(RangeError);
  });

  it("half-even 后统一对齐到 17k+5", () => {
    expect(alignH3FrameCount(146)).toBe(158);
    expect(alignH3FrameCount(520)).toBe(532);
    expect(alignH3Frames(5, 24)).toBe(124);
    expect(alignH3Frames(5.5, 1)).toBe(22); // odd 5 rounds up to 6
    expect(alignH3Frames(22.5, 1)).toBe(22); // even 22 stays 22
    expect(alignH3Frames(39.5, 1)).toBe(56); // odd 39 rounds up to 40
    expect(alignH3Frames(56.5, 1)).toBe(56); // even 56 stays 56
  });

  it("在 512 上限附近保持与后端相同的 half-even 边界", () => {
    expect(alignH3Frames(99.7, 5)).toBe(498); // raw 498.5 -> even 498
    expect(alignH3Frames(99.8, 5)).toBe(515); // raw 499 -> next 17k+5
  });

  it("总有效帧只累计启用镜头，并以帧率换算有效时长", () => {
    const draft = createInitialDrafts().t2v;
    draft.shots.push({ ...draft.shots[0], id: "shot-disabled", enabled: false });
    expect(effectiveShotTiming(5, 24)).toEqual({
      frames: 124,
      durationSeconds: 124 / 24,
    });
    expect(effectiveDraftTiming(draft)).toEqual({
      frames: 124,
      durationSeconds: 124 / 24,
    });
  });
});
