import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  BUNDLED_CATALOGS,
  createTranslator,
  I18nProvider,
  useTranslator,
  type BundledTranslationKey,
  type MessageCatalogs,
} from "../i18n";

function Greeting({ name = "导演" }: { name?: string }) {
  const { locale, t } = useTranslator();
  return <p>{locale}：{t("demo.greeting", { name })}</p>;
}

describe("Director frontend i18n", () => {
  it("读取内置中文文本并替换参数", () => {
    const translator = createTranslator();
    const retryKey: BundledTranslationKey = "common.retry";

    expect(translator.locale).toBe("zh-CN");
    expect(Object.values(BUNDLED_CATALOGS["zh-CN"]).every((value) => typeof value === "string"))
      .toBe(true);
    expect(translator.t(retryKey)).toBe("重试");
    expect(translator.t("errors.model_binding_required.message", {
      bindings: "编码模型（CLIP）、视频编解码模型（Video VAE）",
    })).toBe("当前项目缺少编码模型（CLIP）、视频编解码模型（Video VAE）。");
    expect(translator.t("errors.segment_frame_limit_exceeded.message", { maxFrames: 512 }))
      .toBe("所选片段超过 MiniMax H3 的 512 帧生成上限。");
  });

  it("当前语言缺少文本时回退中文，再回退调用方文本或稳定 key", () => {
    const catalogs: MessageCatalogs = {
      "en-US": { "demo.greeting": "Hello, {name}!" },
    };
    const translator = createTranslator({ locale: "en-US", catalogs });

    expect(translator.t("demo.greeting", { name: "Director" })).toBe("Hello, Director!");
    expect(translator.t("common.retry")).toBe("重试");
    expect(translator.t("missing.key", undefined, "可读回退")).toBe("可读回退");
    expect(translator.t("still.missing")).toBe("still.missing");
    expect(translator.t("demo.greeting")).toBe("Hello, {name}!");
  });

  it("Provider 支持未来语言切换，未挂 Provider 时仍使用内置中文", () => {
    const catalogs: MessageCatalogs = {
      "en-US": { "demo.greeting": "Hello, {name}!" },
    };
    const standalone = render(<Greeting />);
    expect(screen.getByText("zh-CN：demo.greeting")).toBeInTheDocument();
    standalone.unmount();

    const view = render(
      <I18nProvider locale="en-US" catalogs={catalogs}>
        <Greeting name="Director" />
      </I18nProvider>,
    );
    expect(screen.getByText("en-US：Hello, Director!")).toBeInTheDocument();

    view.rerender(
      <I18nProvider locale="zh-CN" catalogs={catalogs}>
        <Greeting />
      </I18nProvider>,
    );
    expect(screen.getByText("zh-CN：demo.greeting")).toBeInTheDocument();
  });
});
