import styles from "../styles.css?raw";

describe("顶栏项目控件焦点样式", () => {
  it("保留全局焦点环，并把项目控件的轮廓完整收进裁切容器内", () => {
    const globalFocusRule = styles.match(/button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible\s*\{([^}]*)\}/)?.[1];
    const containedFocusRule = styles.match(/\.topbar__mode--timeline>:is\(button,input,select\):focus-visible\s*\{([^}]*)\}/)?.[1];

    expect(globalFocusRule).toMatch(/outline\s*:\s*2px\s+solid/);
    expect(containedFocusRule).toMatch(/outline-offset\s*:\s*-3px/);
  });

  it("为时间线项目控件提供完整的浅色主题表面与文字颜色", () => {
    const titleRule = styles.match(/:root\[data-theme="light"\] \.topbar--timeline \.topbar__project-title\{([^}]*)\}/)?.[1];
    const switcherRule = styles.match(/:root\[data-theme="light"\] \.topbar__project-switcher\{([^}]*)\}/)?.[1];
    const deleteRule = styles.match(/:root\[data-theme="light"\] \.topbar__project-delete\{([^}]*)\}/)?.[1];
    const deleteHoverRule = styles.match(/:root\[data-theme="light"\] \.topbar__project-delete:hover\{([^}]*)\}/)?.[1];

    expect(titleRule).toMatch(/color\s*:\s*#3f3832/);
    expect(switcherRule).toMatch(/background\s*:\s*#fffaf3/);
    expect(switcherRule).toMatch(/color\s*:\s*#302b27/);
    expect(deleteRule).toMatch(/background\s*:\s*#fae9e5/);
    expect(deleteRule).toMatch(/color\s*:\s*#9f4039/);
    expect(deleteHoverRule).toMatch(/background\s*:\s*#f6d7d1/);
  });
});

describe("数字输入框", () => {
  it("在 Firefox 和 WebKit 中都隐藏原生增减按钮", () => {
    const numberRule = styles.match(/input\[type="number"\]\{([^}]*)\}/)?.[1];
    const webkitRule = styles.match(/input\[type="number"\]::\-webkit-inner-spin-button,input\[type="number"\]::\-webkit-outer-spin-button\{([^}]*)\}/)?.[1];

    expect(numberRule).toMatch(/appearance\s*:\s*textfield/);
    expect(numberRule).toMatch(/-moz-appearance\s*:\s*textfield/);
    expect(webkitRule).toMatch(/-webkit-appearance\s*:\s*none/);
  });
});

describe("源视频卡响应式布局", () => {
  it("侧边栏挤压工作区时仍保持素材和设置并排", () => {
    const sourceLayoutRule = styles.match(/\.segment-reference-grid__source-layout\{([^}]*grid-template-columns:repeat\(2,minmax\(0,1fr\)\)[^}]*)\}/)?.[1];
    const settingsRule = styles.match(/\.segment-reference-grid__settings \.segment-source-range\{([^}]*display:grid[^}]*)\}/)?.[1];
    const fieldRule = styles.match(/\.segment-reference-grid__settings \.segment-source-range__field\{([^}]*)\}/)?.[1];

    expect(styles).not.toMatch(/@container \(min-width:340px\)/);
    expect(sourceLayoutRule).toMatch(/gap\s*:\s*5px/);
    expect(settingsRule).toMatch(/grid-template-columns\s*:\s*1fr/);
    expect(fieldRule).toMatch(/grid-template-columns\s*:\s*1fr/);
  });

  it("源视频缩略图宽度与参考素材的两列和三列网格一致", () => {
    const threeColumnRule = styles.match(/@container \(min-width:521px\)\{([\s\S]*?)\n\}/)?.[1];

    expect(threeColumnRule).toMatch(/grid-template-columns:calc\(\(100% - 10px\)\/3\) minmax\(0,1fr\)/);
  });

  it("只在较宽卡片中让两个数值项同行", () => {
    const wideRule = styles.match(/@container \(min-width:620px\)\{([\s\S]*?)\n\}/)?.[1];

    expect(wideRule).toMatch(/\.segment-reference-grid__settings \.segment-source-range\{grid-template-columns:repeat\(2,minmax\(0,1fr\)\)\}/);
    expect(wideRule).toMatch(/\.source-audio-reference-toggle\{grid-column:1\/-1\}/);
  });
});
