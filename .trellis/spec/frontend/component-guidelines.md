# Component & Styling Guidelines

> How components, design tokens, and UI elements are styled and structured in the MacBERT Interactive Workbench.

---

## Overview

The Workbench adopts a **Pure Minimalist Monochrome** (Linear / Vercel style) design system. All visual elements, states, badges, and charts are expressed strictly within a black-and-white / grayscale spectrum without chromatic hue saturation, ensuring high academic rigor, visual calm, and modern aesthetics.

---

## Design System & Color Palette

### 1. Grayscale Palette (Pure Grayscale Constraint: R = G = B)
- **Extreme Contrast**:
  - Dark Theme: Background `#000000` / `#0a0a0a`, Surface `#141414`, Text Main `#ffffff`, Text Secondary `#a3a3a3`, Text Muted `#737373`
  - Light Theme: Background `#ffffff`, Surface `#fafafa`, Text Main `#000000`, Text Secondary `#525252`, Text Muted `#8c8c8c`
- **Zero Saturated Chroma**:
  - Colored hex codes (`#3b82f6`, `#10b981`, `#8b5cf6`, `#ef4444`, etc.) are forbidden in component CSS, status chips, and SVG charts.
  - Success / Match states use `#ffffff` (Dark) / `#000000` (Light) with clean checkmark icons (`○`, `✓`).
  - Discrepancy / Diff states use solid neutral cards with left borders (`border-left: 3px solid var(--text-main)`) and discrete indicator glyphs (`●`).

---

## 10 CLUENER Entity Category Differentiation Contract

To maintain distinct readability among 10 entity categories without colors, components use structural CSS differentiation (line styles, lightness scales, inversions, and font-weights):

| Category | CN Name | Key Visual Distinctions | Dark Mode Token | Light Mode Token |
|---|---|---|---|---|
| `name` | 姓名 | Inverted high-contrast pill | Pure White bg, Black text, bold | Pure Black bg, White text, bold |
| `government` | 政府机构 | Solid heavy border | `#ffffff` solid 1px | `#000000` solid 1px |
| `company` | 公司 | Mid-high solid border & bold | `#d4d4d4` solid, 700 weight | `#262626` solid, 700 weight |
| `organization` | 组织机构 | Subtle border, semi-bold | `#a3a3a3` solid, 600 weight | `#404040` solid, 600 weight |
| `position` | 职位 | Dashed thin border | `#737373` dashed 1px | `#737373` dashed 1px |
| `book` | 书籍 | Dashed medium border | `#e5e5e5` dashed 1px | `#333333` dashed 1px |
| `game` | 游戏 | Dotted border | `#d4d4d4` dotted 1px | `#404040` dotted 1px |
| `scene` | 景点 | Double solid border | `#ffffff` double 3px | `#000000` double 3px |
| `address` | 地址 | Soft grayscale outline | `#525252` solid 1px | `#a3a3a3` solid 1px |
| `movie` | 电影 | Muted subtle tone | `#404040` solid 1px | `#8c8c8c` solid 1px |

---

## Model Distinction Patterns

- **Chinese-MacBERT (Proposed)**:
  - Visual Dominance: High-contrast pure white (dark mode) / pure black (light mode).
  - Accent indicators: Solid lines, primary filled buttons, prominent labels.
- **BERT-base (Baseline)**:
  - Visual Subordination: Muted mid-gray (`#737373` / `#525252`).
  - Secondary badges and standard weight typography.

---

## SVG Chart Conventions

- **SVG Gridlines**: Grayscale dashed lines (`stroke: #262626` in dark mode, `#e5e5e5` in light mode).
- **Bar Styling**:
  - MacBERT bars: High-contrast `fill="var(--macbert-bar)"` (`#ffffff` dark / `#000000` light).
  - BERT bars: Subdued `fill="var(--bert-bar)"` (`#525252` dark / `#a3a3a3` light).
  - Delta callouts: Dynamic high contrast percentage indicators.
- **Theme Reactivity**: `setTheme(theme)` must synchronously invoke `renderBenchmarkChart()` to re-render SVG shapes with the matching theme palette.

---

## Common Mistakes

- Adding RGB chroma for success/warning alerts instead of utilizing typography, borders, and monochrome glyphs.
- Forgetting to escape single braces (`{{` and `}}`) in Python `build_workbench_html.py` template string.
- Using hardcoded colors in dynamic JS entity chip rendering instead of referencing `badge-${category}` classes.
