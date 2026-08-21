import { Component, type PropsWithChildren } from "react";

interface RootErrorBoundaryState {
  hasError: boolean;
}

export class RootErrorBoundary extends Component<PropsWithChildren, RootErrorBoundaryState> {
  state: RootErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): RootErrorBoundaryState {
    return { hasError: true };
  }

  private retry = () => {
    this.setState({ hasError: false });
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <main className="root-error-boundary">
        <section className="panel root-error-boundary__panel" role="alert" aria-labelledby="root-error-title">
          <span className="eyebrow">Director</span>
          <h1 id="root-error-title">页面发生错误</h1>
          <p>Director 无法继续显示当前页面。你可以重试；如果问题仍然存在，请重新加载页面。</p>
          <div className="root-error-boundary__actions">
            <button className="button button--primary" type="button" onClick={this.retry}>重试</button>
            <button className="button button--ghost" type="button" onClick={() => window.location.reload()}>重新加载页面</button>
          </div>
        </section>
      </main>
    );
  }
}
