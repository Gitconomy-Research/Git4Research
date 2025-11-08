# Gitconomy社区信息图设计与实现指南

## 1. 引言

这份设计与实现指南是[《Gitconomy社区图形即代码设计规范指南》](./Gitconomy社区图形即代码设计规范指南.md)的补充与具体应用，旨在为Gitconomy社区的开发者与内容贡献者在编写文档、课程讲义或研究报告时，高效地创建专业、一致且富有表现力的信息图（Infographic）与其他可视化内容，从而显著增强文档信息的可读性与传达效果。

---

## 2. 核心设计原则

  * **一致性：** 所有图形元素（色彩、字体、间距）必须遵循统一的设计系统，确保品牌形象的统一。
  * **清晰性：** 设计应服务于信息传达，通过清晰的视觉层级，引导用户快速理解复杂内容。
  * **可维护性：** 通过使用CSS变量和组件化思想，使设计规范的更新和维护变得简单高效。
  * **可访问性：** 确保图形内容能被包括屏幕阅读器在内的辅助技术所理解，惠及所有用户。

---

## 3. 核心技术栈

  * **结构：** HTML5
  * **样式：** CSS3 (重点使用CSS变量)
  * **图形：** SVG (内联于HTML中)

---

## 4. 设计系统实现

设计系统的核心在于将颜色、字体等设计元素组件化，并通过CSS变量进行全局管理。这使得所有视觉元素都源自单一可信来源，便于维护和主题切换（如未来增加深色模式）。

### 4.1 调色板

所有颜色值都应定义在 `:root` 伪类中，以便全局复用。

**代码样例：**

```css
/* 在 <style> 标签中定义 */
:root {
    /* 填充色 (Fill) */
    --color-main-box-fill: #EFF6FF;
    --color-box-fill: #F0FDF4;
    --color-card-background-fill: #DBEAFE;
    --color-highlight-box-fill: #FEF9C3;
    --color-background: white;

    /* 描边/强调色 (Stroke/Accent) */
    --color-main-box-stroke: #60A5FA;
    --color-box-stroke: #34D399;
    --color-card-background-stroke: #3B82F6;
    --color-highlight-box-stroke: #FACC15;
    --color-connector-standard: #2563EB;
    --color-connector-highlight: #FACC15;

    /* 文本颜色 (Text) */
    --color-text-main-title: #111827;
    --color-text-body: #374151;
    --color-text-small: #6B7280;
    --color-text-license: #64748B;
}
```

### 4.2 字体规范

字体相关的属性，如字族、大小、字重，也应通过CSS变量进行管理。

**代码样例：**

```css
/* 在 <style> 标签中定义 */
:root {
    /* ... 颜色变量 ... */

    /* 字体族 */
    --font-family-sans: 'Noto Sans', 'Inter', sans-serif;

    /* 字号 */
    --font-size-main-title: 22px;
    --font-size-title: 16px;
    --font-size-body: 12px;
    --font-size-small: 11px;
    --font-size-license: 12px;

    /* 字重 */
    --font-weight-main-title: 800; /* Extra Bold */
    --font-weight-bold: 700;
    --font-weight-normal: 400;
}
```

---

## 5. SVG实现最佳实践

### 5.1 基础结构与可访问性

每个内联SVG都应被视为一幅图像，并为其提供必要的辅助信息。

  * **`viewBox` 属性：** 必须设置，它定义了SVG的内部坐标系，是实现响应式缩放的关键。
  * **`role="img"`：** 明确向辅助技术声明该SVG元素是一张图片。
  * **`<title>` 和 `<desc>`：** 作为SVG的**首要子元素**，提供图像的标题和详细描述。`aria-labelledby` 属性用于将这些描述与SVG本身关联起来。

### 5.2 使用CSS对SVG元素进行样式化

在HTML的`<style>`块中定义样式类，然后在SVG内部元素的`class`属性中引用这些类。这是连接设计系统和图形实现的关键。

### 5.3 源代码注释

为复杂的SVG或其逻辑分组添加注释，是保持代码可读性和可维护性的重要环节。

---

## 6. 完整实现范例

以下是一个完整的[信息图示例代码](./../assets/gitconomy-community-infographic-design-guideline-example.html)，它集成了上述所有实践。

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gitconomy社区信息图实现范例</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;800&family=Inter:wght@400;500;700;800&display=swap" rel="stylesheet">

    <style>
        /* 全局样式与布局 */
        body {
            font-family: 'Noto Sans SC', 'Inter', sans-serif;
            background-color: #f9fafb;
            color: #111827;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .infographic-container {
            max-width: 800px;
            width: 100%;
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            padding: 2rem;
        }

        /* 1. 设计系统: CSS变量 (Design Tokens) */
        :root {
            /* 颜色变量 */
            --color-main-box-fill: #EFF6FF;
            --color-main-box-stroke: #60A5FA;
            --color-card-background-fill: #DBEAFE;
            --color-card-background-stroke: #3B82F6;
            --color-highlight-box-fill: #FEF9C3;
            --color-highlight-box-stroke: #FACC15;
            --color-connector-standard: #2563EB;
            --color-text-main-title: #111827;
            --color-text-section-title: #374151;
            --color-text-card-title: #1E40AF;
            --color-text-body: #374151;
            --color-text-license: #64748B;
            --color-icon: #3B82F6;
            --color-data-viz-track: #E5E7EB;
            --color-data-viz-progress: var(--color-main-box-stroke);

            /* 字体变量 */
            --font-family-sans: 'Noto Sans SC', 'Inter', sans-serif;
            --font-size-main-title: 24px;
            --font-size-section-title: 18px;
            --font-size-card-title: 14px;
            --font-size-body: 12px;
            --font-size-license: 12px;
            --font-weight-main-title: 800;
            --font-weight-bold: 700;
            --font-weight-medium: 500;
            --font-weight-normal: 400;
        }

        /* 2. SVG及HTML元素样式类定义 */
        .card-background { fill: var(--color-card-background-fill); stroke: var(--color-card-background-stroke); stroke-width: 1.5px; }
        .elbow-connector { fill: none; stroke: var(--color-connector-standard); stroke-width: 2px; }
        .icon-style { fill: var(--color-icon); }

        /* 文本样式 */
        .section-title-text { font-family: var(--font-family-sans); font-size: var(--font-size-section-title); font-weight: var(--font-weight-bold); fill: var(--color-text-section-title); }
        .card-title-text { font-family: var(--font-family-sans); font-size: var(--font-size-card-title); font-weight: var(--font-weight-bold); fill: var(--color-text-card-title); }
        .body-text { font-family: var(--font-family-sans); font-size: var(--font-size-body); font-weight: var(--font-weight-normal); fill: var(--color-text-body); }

        /* 数据可视化样式 */
        .data-viz-track { fill: none; stroke: var(--color-data-viz-track); }
        .data-viz-progress { fill: none; stroke: var(--color-data-viz-progress); transition: stroke-dashoffset 0.5s ease-in-out; }
        .data-viz-text { font-family: var(--font-family-sans); font-size: 24px; font-weight: var(--font-weight-main-title); fill: var(--color-text-main-title); text-anchor: middle; }

        /* HTML元素样式 */
        .main-title { font-family: var(--font-family-sans); font-size: var(--font-size-main-title); font-weight: var(--font-weight-main-title); color: var(--color-text-main-title); text-align: center; margin-bottom: 2rem; }
        .license-text { font-family: var(--font-family-sans); font-size: var(--font-size-license); color: var(--color-text-license); text-align: center; margin-top: 2rem; }
        svg { width: 100%; height: auto; }

    </style>
</head>

<body>

    <div class="infographic-container">

        <header>
            <h1 class="main-title">Gitconomy社区信息图实现范例</h1>
        </header>

        <main>
            <svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="diagramTitle diagramDesc">

                <title id="diagramTitle">Gitconomy社区信息图实现范例</title>
                <desc id="diagramDesc">此图分为两部分。左侧展示了从“规划设计”到“开发实现”再到“审查合并”的三步工作流程。右侧通过一个环形图和列表展示了项目的关键特性。</desc>

                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="var(--color-connector-standard)" />
                    </marker>
                </defs>

                <g id="workflow-section">
                    <text x="180" y="50" class="section-title-text" text-anchor="middle">核心三步流程</text>

                    <g transform="translate(50, 100)">
                        <rect width="260" height="100" rx="8" class="card-background"/>
                        <circle cx="40" cy="50" r="20" fill="white"/>
                        <text x="40" y="56" font-size="18" font-weight="bold" fill="var(--color-icon)" text-anchor="middle">1</text>
                        <g class="icon-style" transform="translate(80, 35) scale(1.2)">
                            <path d="M12.9 6.1a1 1 0 0 0-1.8 0l-1.1 2.3-2.5.4a1 1 0 0 0-.6 1.7l1.8 1.8-.4 2.5a1 1 0 0 0 1.5 1.1l2.2-1.2 2.2 1.2a1 1 0 0 0 1.5-1.1l-.4-2.5 1.8-1.8a1 1 0 0 0-.6-1.7l-2.5-.4-1.1-2.3z"/>
                        </g>
                        <text x="110" y="45" class="card-title-text">规划设计</text>
                        <text x="110" y="65" class="body-text">明确需求，制定技术方案。</text>
                    </g>

                    <g transform="translate(50, 250)">
                        <rect width="260" height="100" rx="8" class="card-background"/>
                        <circle cx="40" cy="50" r="20" fill="white"/>
                        <text x="40" y="56" font-size="18" font-weight="bold" fill="var(--color-icon)" text-anchor="middle">2</text>
                        <g class="icon-style" transform="translate(80, 35) scale(1.2)">
                           <path d="M14.2 9.8a1 1 0 0 1 1.6 1.2l-2.4 4.5a1 1 0 0 1-1.6-1.2l2.4-4.5zM4.2 9.8a1 1 0 0 0 1.6 1.2l2.4-4.5a1 1 0 0 0-1.6-1.2L4.2 9.8zM12 3a1 1 0 0 1 1 1v2a1 1 0 1 1-2 0V4a1 1 0 0 1 1-1z"/>
                        </g>
                        <text x="110" y="45" class="card-title-text">开发实现</text>
                        <text x="110" y="65" class="body-text">编写高质量、可维护的代码。</text>
                    </g>

                    <g transform="translate(50, 400)">
                        <rect width="260" height="100" rx="8" class="card-background"/>
                        <circle cx="40" cy="50" r="20" fill="white"/>
                        <text x="40" y="56" font-size="18" font-weight="bold" fill="var(--color-icon)" text-anchor="middle">3</text>
                        <g class="icon-style" transform="translate(80, 35) scale(1.2)">
                           <path d="M16.3 5.7a1 1 0 0 0-1-1.6l-4 2a1 1 0 0 0-1 .6l-3 7a1 1 0 0 0 1.6 1l3-7-2.4 1.2a1 1 0 1 0 1 1.6l4-2a1 1 0 0 0 .4-1.6l-3-7 2.4-1.2z"/>
                        </g>
                        <text x="110" y="45" class="card-title-text">审查合并</text>
                        <text x="110" y="65" class="body-text">通过同行评审确保代码质量。</text>
                    </g>

                    <path d="M180,200 v 30" class="elbow-connector" marker-end="url(#arrowhead)"/>
                    <path d="M180,350 v 30" class="elbow-connector" marker-end="url(#arrowhead)"/>
                </g>

                <g id="features-section" transform="translate(400, 0)">
                    <text x="200" y="50" class="section-title-text" text-anchor="middle">关键特性解析</text>

                    <g transform="translate(200, 180)">
                        <circle cx="0" cy="0" r="60" stroke-width="20" class="data-viz-track" />
                        <circle cx="0" cy="0" r="60" stroke-width="20" class="data-viz-progress"
                                transform="rotate(-90)"
                                stroke-dasharray="377"
                                stroke-dashoffset="94.25" />
                        <text y="10" class="data-viz-text">75%</text>
                    </g>
                    <text x="200" y="280" text-anchor="middle" class="body-text">社区贡献率</text>

                    <g transform="translate(80, 350)">
                        <text x="0" y="0" class="section-title-text">核心优势</text>

                        <circle cx="10" cy="40" r="3" fill="var(--color-icon)"/>
                        <text x="25" y="45" class="body-text">开放透明的治理模式</text>

                        <circle cx="10" cy="70" r="3" fill="var(--color-icon)"/>
                        <text x="25" y="75" class="body-text">代码即价值的贡献账本</text>

                        <circle cx="10" cy="100" r="3" fill="var(--color-icon)"/>
                        <text x="25" y="105" class="body-text">自动化的激励分配机制</text>

                        <circle cx="10" cy="130" r="3" fill="var(--color-icon)"/>
                        <text x="25" y="135" class="body-text">高度可扩展的插件架构</text>
                    </g>
                </g>
            </svg>
        </main>
    </div>

    <footer>
        <p class="license-text">
            本作品采用CC-BY-SA 4.0国际许可协议进行许可，© 2025 Gitconomy Research社区
        </p>
    </footer>

</body>
</html>
```

---

## 附录：术语表

### CSS (层叠样式表)

| 术语 (Term) | 说明 (Description) |
| :--- | :--- |
| **`:root`** | 一个CSS伪类，代表文档的根元素。在本指南中，它被用作一个中心化的位置，来声明所有全局的CSS变量（如`--color-main-box-fill`），从而建立整个设计系统的“单一可信来源”。 |
| **`fill`** | 一个CSS属性，用于设置SVG图形（如`<rect>`, `<circle>`, `<path>`）的**内部填充颜色**。它不影响描边。 |
| **`stroke`** | 一个CSS属性，用于设置SVG图形的**描边（轮廓）颜色**。 |
| **`stroke-dasharray`** | 一个CSS属性，它将图形的描边变为一系列的虚线。在本指南中，它与`stroke-dashoffset`结合使用，通过精确计算圆形的周长来实现**环形图**的进度效果。 |
| **`stroke-dashoffset`** | 一个CSS属性，用于设定`stroke-dasharray`所创建的虚线的起始**偏移量**。通过动态改变这个值，可以创建进度条动画。 |
| **`stroke-width`** | 一个CSS属性，定义SVG图形描边的**宽度（粗细）**。 |
| **`text-anchor`** | 一个SVG特有的CSS属性，用于定义文本相对于其定位点（x, y坐标）的对齐方式。范例中`text-anchor="middle"`用于使文本**水平居中**。 |
| **`var()`** | 一个CSS函数，用于**调用**在`:root`中定义的CSS变量。例如，`fill: var(--color-main-box-fill);`会读取变量值并应用到`fill`属性上。 |

### SVG (可缩放矢量图形)

| 术语 (Term) | 说明 (Description) |
| :--- | :--- |
| **`<svg>`** | SVG图形的**根元素**，相当于整个图形的画布。所有SVG图形内容都必须包裹在此标签内。 |
| **`<circle>`** | 一个基本的SVG图形元素，用于绘制**圆形**。在范例中用于数据环形图的轨道和进度条，以及特性列表前的项目符号。 |
| **`<defs>`** | SVG中的“定义”元素，用于容纳那些**可被重复使用的图形对象**，如`<marker>`。它内部的元素在定义时不会被直接渲染。 |
| **`<desc>`** | “描述”元素，与`<title>`配合使用，为辅助技术（如屏幕阅读器）提供关于SVG内容的更详细的、非结构化的**文字描述**。 |
| **`<g>`** | “分组”元素，用于将相关的图形元素组合成一个逻辑单元或**“组件”**。对`<g>`标签应用`transform`可以移动整个组件。 |
| **`<marker>`** | 定义在`<defs>`中的元素，用于为`<path>`或`<line>`的顶点添加如图形、**箭头**等符号。 |
| **`<path>`** | 最强大的SVG图形元素之一，通过`d`属性中的一系列命令来绘制任意形状的路径。在范例中用于绘制**连接线**和自定义**图标**。 |
| **`<rect>`** | 一个基本的SVG图形元素，用于绘制**矩形**。在范例中主要用于创建卡片的背景。 |
| **`<text>`** | 用于在SVG画布内创建**文本**。其样式（颜色、字体）可以通过CSS类进行控制。 |
| **`<title>`** | 为SVG图形提供一个可访问的、人类可读的**标题**。它通常是SVG的第一个子元素，对SEO和可访问性至关重要。 |
| **`transform`** | 一个SVG属性（不是标签），用于对元素或元素组`<g>`进行移动(`translate`)、缩放(`scale`)、旋转(`rotate`)等几何变换。 |
| **`viewBox`** | `<svg>`元素的一个关键属性，它定义了画布的内部坐标系和宽高比。`viewBox`是实现图形**响应式**的核心机制。 |

---

## 许可声明

本文档采用[知识共享署名--相同方式共享4.0国际许可协议(CC BY--SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.zh)进行许可，&copy; 2025 Gitconomy Research社区
