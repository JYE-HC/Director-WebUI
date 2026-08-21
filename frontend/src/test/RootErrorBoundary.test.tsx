import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RootErrorBoundary } from "../components/RootErrorBoundary";

describe("RootErrorBoundary", () => {
  it("shows a generic fallback and retries the child tree", async () => {
    const user = userEvent.setup();
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    let shouldThrow = true;

    function FlakyView() {
      if (shouldThrow) throw new Error("private implementation detail");
      return <p>页面已恢复</p>;
    }

    try {
      render(<RootErrorBoundary><FlakyView /></RootErrorBoundary>);

      expect(screen.getByRole("alert", { name: "页面发生错误" })).toBeInTheDocument();
      expect(screen.queryByText(/private implementation detail/)).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "重新加载页面" })).toBeInTheDocument();

      shouldThrow = false;
      await user.click(screen.getByRole("button", { name: "重试" }));

      expect(screen.getByText("页面已恢复")).toBeInTheDocument();
    } finally {
      consoleError.mockRestore();
    }
  });
});
