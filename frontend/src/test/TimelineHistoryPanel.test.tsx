import { createRef } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { TimelineHistoryPanel } from "../components/TimelineHistoryPanel";
import { createTimelineProject } from "../domain/timelineProject";
import {
  createTimelineHistory,
  recordTimelineHistory,
  undoTimelineHistory,
} from "../state/timelineHistory";

function historyFixture() {
  const initial = createTimelineProject();
  const renamed = { ...initial, title: "新项目名" };
  const edited = {
    ...renamed,
    segments: [{ ...renamed.segments[0], prompt: "镜头提示词" }],
  };
  let history = recordTimelineHistory(createTimelineHistory(), {
    label: "重命名项目",
    before: initial,
    after: renamed,
  });
  history = recordTimelineHistory(history, {
    label: "编辑提示词",
    before: renamed,
    after: edited,
  });
  return history;
}

describe("TimelineHistoryPanel", () => {
  it("显示当前位置、已应用与已撤销状态，并按游标请求一次跳转", () => {
    const onJump = vi.fn();
    const history = undoTimelineHistory(historyFixture())!.history;
    render(<TimelineHistoryPanel
      id="history-panel"
      open
      history={history}
      toggleRef={createRef<HTMLButtonElement>()}
      onJump={onJump}
      onClose={vi.fn()}
    />);

    expect(screen.getByText("当前位置 1 / 2")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重命名项目，当前位置" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "编辑提示词，已撤销" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "编辑提示词，已撤销" }));
    expect(onJump).toHaveBeenCalledTimes(1);
    expect(onJump).toHaveBeenCalledWith(2);
  });

  it("Escape 请求关闭并回焦，外部点击关闭但不抢焦", () => {
    const onClose = vi.fn();
    const toggleRef = createRef<HTMLButtonElement>();
    render(<>
      <button ref={toggleRef}>编辑历史</button>
      <TimelineHistoryPanel
        id="history-panel"
        open
        history={historyFixture()}
        toggleRef={toggleRef}
        onJump={vi.fn()}
        onClose={onClose}
      />
      <button>外部区域</button>
    </>);

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenLastCalledWith(true);
    fireEvent.pointerDown(screen.getByRole("button", { name: "外部区域" }));
    expect(onClose).toHaveBeenLastCalledWith(false);
  });

  it("关闭时保留 DOM 关系但不暴露面板内容", () => {
    render(<TimelineHistoryPanel
      id="history-panel"
      open={false}
      history={createTimelineHistory()}
      toggleRef={createRef<HTMLButtonElement>()}
      onJump={vi.fn()}
      onClose={vi.fn()}
    />);

    expect(document.getElementById("history-panel")).toHaveAttribute("hidden");
    expect(screen.queryByRole("complementary", { name: "编辑历史" })).not.toBeInTheDocument();
  });
});
