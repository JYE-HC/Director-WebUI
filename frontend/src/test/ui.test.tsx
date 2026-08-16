import { useState, type FormEvent } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DeferredNumberInput } from "../components/ui";

function NumberHarness({ onCommit }: { onCommit: (value: number) => void }) {
  const [value, setValue] = useState(4);
  return <DeferredNumberInput
    aria-label="最短帧"
    min={4}
    max={100}
    step={1}
    value={value}
    normalizeValue={Math.trunc}
    onValueCommit={(next) => {
      setValue(next);
      onCommit(next);
    }}
  />;
}

describe("延迟提交数字输入框", () => {
  it("允许清空和输入临时低于下限的前缀，失焦后才提交完整值", async () => {
    const user = userEvent.setup();
    const onCommit = vi.fn();
    render(<NumberHarness onCommit={onCommit} />);
    const input = screen.getByLabelText("最短帧") as HTMLInputElement;

    await user.clear(input);
    expect(input.value).toBe("");
    await user.type(input, "12");
    expect(input.value).toBe("12");
    expect(onCommit).not.toHaveBeenCalled();

    await user.tab();
    expect(onCommit).toHaveBeenCalledOnce();
    expect(onCommit).toHaveBeenLastCalledWith(12);
    expect(input).toHaveValue(12);
  });

  it("仅在失焦或 Enter 时夹紧上下限，空值失焦恢复原值", async () => {
    const user = userEvent.setup();
    const onCommit = vi.fn();
    render(<NumberHarness onCommit={onCommit} />);
    const input = screen.getByLabelText("最短帧") as HTMLInputElement;

    await user.clear(input);
    await user.tab();
    expect(input).toHaveValue(4);
    expect(onCommit).not.toHaveBeenCalled();

    await user.clear(input);
    await user.type(input, "1");
    expect(input).toHaveValue(1);
    expect(onCommit).not.toHaveBeenCalled();
    await user.keyboard("{Enter}");
    expect(input).toHaveValue(4);
    expect(onCommit).not.toHaveBeenCalled();

    await user.click(input);
    await user.clear(input);
    await user.type(input, "101");
    await user.tab();
    expect(onCommit).toHaveBeenLastCalledWith(100);
    expect(input).toHaveValue(100);
  });

  it("保留负数和小数的逐键输入过程", async () => {
    const user = userEvent.setup();
    const onCommit = vi.fn();
    render(<DeferredNumberInput
      aria-label="LoRA 强度"
      min={-10}
      max={10}
      step={0.01}
      value={0}
      onValueCommit={onCommit}
    />);
    const input = screen.getByLabelText("LoRA 强度") as HTMLInputElement;

    await user.clear(input);
    await user.type(input, "-1.25");
    expect(input.value).toBe("-1.25");
    expect(onCommit).not.toHaveBeenCalled();
    await user.tab();
    expect(onCommit).toHaveBeenCalledWith(-1.25);
  });

  it("Enter 只提交当前数值，不误提交外层表单", async () => {
    const user = userEvent.setup();
    const onCommit = vi.fn();
    const onSubmit = vi.fn((event: FormEvent) => event.preventDefault());
    render(<form onSubmit={onSubmit}>
      <DeferredNumberInput
        aria-label="步数"
        min={1}
        max={200}
        step={1}
        value={25}
        normalizeValue={Math.trunc}
        onValueCommit={onCommit}
      />
    </form>);

    const input = screen.getByLabelText("步数");
    await user.clear(input);
    await user.type(input, "31");
    await user.keyboard("{Enter}");
    expect(onCommit).toHaveBeenCalledOnce();
    expect(onCommit).toHaveBeenCalledWith(31);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("父级将提交值纠正回原权威值时不会残留本地草稿", async () => {
    const user = userEvent.setup();
    const onCommit = vi.fn();
    render(<DeferredNumberInput
      aria-label="源截取时长"
      min={0.01}
      value={39 / 24}
      onValueCommit={onCommit}
    />);
    const input = screen.getByLabelText("源截取时长");

    await user.clear(input);
    await user.type(input, "1{Enter}");

    expect(onCommit).toHaveBeenCalledOnce();
    expect(onCommit).toHaveBeenCalledWith(1);
    expect(input).toHaveValue(39 / 24);
  });
});
