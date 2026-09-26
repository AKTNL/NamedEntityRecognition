"""
Workbench HTML & Template Builder
生成兼具现代学术科技高颜值 UI、完全自包含（零外部网络依赖）及离线平滑降级的 index.html 与 workbench.html。
"""

import os
import json

ROOT = os.path.dirname(os.path.abspath(__file__))

def build():
    bundle_path = os.path.join(ROOT, "static_offline_bundle.json")
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle_json_str = f.read()

    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MacBERT × CLUENER2020 交互式复现工作台与全栈模型体验系统</title>
  <style>
    /* ==========================================================================
       1. 全局设计系统与主题变量（Pure Minimalist Monochrome / 黑白极简）
       ========================================================================== */
    :root {{
      --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      
      /* Dark Theme (Pure Grayscale / 黑白极简深色) */
      --bg-body: #0a0a0a;
      --bg-navbar: rgba(10, 10, 10, 0.85);
      --bg-card: #141414;
      --bg-card-subtle: #1c1c1c;
      --bg-card-hover: #262626;
      --bg-input: #0e0e0e;
      --border-main: #262626;
      --border-subtle: #383838;
      --border-focus: #ffffff;
      
      --text-main: #ffffff;
      --text-secondary: #a3a3a3;
      --text-muted: #737373;
      
      --accent-primary: #ffffff;
      --accent-primary-hover: #e5e5e5;
      --accent-primary-fg: #0a0a0a;
      --accent-macbert: #ffffff;
      --accent-bert: #737373;
      --accent-success: #ffffff;
      --accent-warning: #a3a3a3;
      --accent-danger: #d4d4d4;
      
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.5);
      --shadow-md: 0 4px 16px -2px rgba(0, 0, 0, 0.6);
      --shadow-lg: 0 10px 30px -4px rgba(0, 0, 0, 0.8);
      
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --radius-full: 9999px;
      
      --transition-fast: 0.15s ease;
      --transition-normal: 0.25s ease;
    }}

    [data-theme="light"] {{
      /* Light Theme (Clean Minimalist Monochrome / 黑白极简浅色) */
      --bg-body: #ffffff;
      --bg-navbar: rgba(255, 255, 255, 0.9);
      --bg-card: #ffffff;
      --bg-card-subtle: #f5f5f5;
      --bg-card-hover: #ebebeb;
      --bg-input: #ffffff;
      --border-main: #e5e5e5;
      --border-subtle: #d4d4d4;
      --border-focus: #000000;
      
      --text-main: #000000;
      --text-secondary: #525252;
      --text-muted: #8c8c8c;
      
      --accent-primary: #000000;
      --accent-primary-hover: #262626;
      --accent-primary-fg: #ffffff;
      --accent-macbert: #000000;
      --accent-bert: #737373;
      --accent-success: #000000;
      --accent-warning: #525252;
      --accent-danger: #262626;
      
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 16px -2px rgba(0, 0, 0, 0.08);
      --shadow-lg: 0 10px 30px -4px rgba(0, 0, 0, 0.12);
    }}

    /* 10 实体类别的极简黑白灰高辨识度体系 (纯色、边框、明度阶梯与纹理差异) */
    .cat-address, .badge-address {{
      --badge-bg: #1c1c1c;
      --badge-border: #444444;
      --badge-color: #ffffff;
    }}
    .cat-book, .badge-book {{
      --badge-bg: #141414;
      --badge-border: #666666;
      --badge-color: #e5e5e5;
      border-style: dashed !important;
    }}
    .cat-company, .badge-company {{
      --badge-bg: #282828;
      --badge-border: #888888;
      --badge-color: #ffffff;
      font-weight: 700;
    }}
    .cat-game, .badge-game {{
      --badge-bg: #181818;
      --badge-border: #666666;
      --badge-color: #d4d4d4;
      border-style: dotted !important;
    }}
    .cat-government, .badge-government {{
      --badge-bg: #000000;
      --badge-border: #ffffff;
      --badge-color: #ffffff;
    }}
    .cat-movie, .badge-movie {{
      --badge-bg: #1e1e1e;
      --badge-border: #555555;
      --badge-color: #b0b0b0;
    }}
    .cat-name, .badge-name {{
      --badge-bg: #ffffff;
      --badge-border: #ffffff;
      --badge-color: #000000;
      font-weight: 700;
    }}
    .cat-organization, .badge-organization {{
      --badge-bg: #222222;
      --badge-border: #aaaaaa;
      --badge-color: #ffffff;
      font-weight: 700;
    }}
    .cat-position, .badge-position {{
      --badge-bg: #141414;
      --badge-border: #444444;
      --badge-color: #8c8c8c;
      border-style: dashed !important;
    }}
    .cat-scene, .badge-scene {{
      --badge-bg: #252525;
      --badge-border: #777777;
      --badge-color: #f0f0f0;
      border-style: double !important;
      border-width: 3px !important;
    }}

    [data-theme="light"] .cat-address, [data-theme="light"] .badge-address {{
      --badge-bg: #f5f5f5;
      --badge-border: #d4d4d4;
      --badge-color: #111111;
    }}
    [data-theme="light"] .cat-book, [data-theme="light"] .badge-book {{
      --badge-bg: #fafafa;
      --badge-border: #888888;
      --badge-color: #222222;
    }}
    [data-theme="light"] .cat-company, [data-theme="light"] .badge-company {{
      --badge-bg: #e8e8e8;
      --badge-border: #666666;
      --badge-color: #000000;
    }}
    [data-theme="light"] .cat-game, [data-theme="light"] .badge-game {{
      --badge-bg: #f5f5f5;
      --badge-border: #777777;
      --badge-color: #333333;
    }}
    [data-theme="light"] .cat-government, [data-theme="light"] .badge-government {{
      --badge-bg: #ffffff;
      --badge-border: #000000;
      --badge-color: #000000;
    }}
    [data-theme="light"] .cat-movie, [data-theme="light"] .badge-movie {{
      --badge-bg: #f0f0f0;
      --badge-border: #cccccc;
      --badge-color: #555555;
    }}
    [data-theme="light"] .cat-name, [data-theme="light"] .badge-name {{
      --badge-bg: #000000;
      --badge-border: #000000;
      --badge-color: #ffffff;
    }}
    [data-theme="light"] .cat-organization, [data-theme="light"] .badge-organization {{
      --badge-bg: #dedede;
      --badge-border: #222222;
      --badge-color: #000000;
    }}
    [data-theme="light"] .cat-position, [data-theme="light"] .badge-position {{
      --badge-bg: #fafafa;
      --badge-border: #cccccc;
      --badge-color: #555555;
    }}
    [data-theme="light"] .cat-scene, [data-theme="light"] .badge-scene {{
      --badge-bg: #eeeeee;
      --badge-border: #888888;
      --badge-color: #111111;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: var(--font-family);
      background-color: var(--bg-body);
      color: var(--text-main);
      line-height: 1.6;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      transition: background-color var(--transition-normal), color var(--transition-normal);
      overflow-x: hidden;
    }}

    a {{
      color: var(--text-main);
      text-decoration: underline;
    }}
    a:hover {{
      color: var(--text-secondary);
    }}

    /* ==========================================================================
       2. 顶部导航与操作栏
       ========================================================================== */
    .navbar {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: var(--bg-navbar);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-main);
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }}

    .nav-brand {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}

    .nav-logo {{
      width: 36px;
      height: 36px;
      border-radius: var(--radius-md);
      background: #000000;
      border: 1px solid #333333;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #ffffff;
      font-weight: 800;
      font-size: 1.1rem;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }}
    [data-theme="light"] .nav-logo {{
      background: #000000;
      border: 1px solid #000000;
      color: #ffffff;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}

    .nav-title-group h1 {{
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .nav-title-group .sub-title {{
      font-size: 0.75rem;
      color: var(--text-muted);
      font-weight: 400;
    }}

    .nav-badge {{
      display: inline-block;
      font-size: 0.65rem;
      padding: 0.15rem 0.45rem;
      border-radius: var(--radius-full);
      font-weight: 600;
      background: var(--bg-card-subtle);
      color: var(--text-secondary);
      border: 1px solid var(--border-main);
    }}

    .nav-actions {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}

    .status-badge {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.75rem;
      padding: 0.35rem 0.75rem;
      border-radius: var(--radius-full);
      font-weight: 500;
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
    }}

    .status-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background-color: var(--text-muted);
    }}
    .status-dot.online {{
      background-color: var(--text-main);
      box-shadow: 0 0 8px var(--text-main);
      animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
      0% {{ transform: scale(0.95); opacity: 0.8; }}
      50% {{ transform: scale(1.15); opacity: 1; }}
      100% {{ transform: scale(0.95); opacity: 0.8; }}
    }}

    .theme-toggle-btn {{
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
      color: var(--text-main);
      width: 36px;
      height: 36px;
      border-radius: var(--radius-md);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background-color var(--transition-fast), border-color var(--transition-fast);
    }}
    .theme-toggle-btn:hover {{
      background: var(--bg-card-hover);
      border-color: var(--border-subtle);
    }}

    /* ==========================================================================
       3. 核心选项卡切换器 (Tabs)
       ========================================================================== */
    .tab-nav-container {{
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-main);
      padding: 0.25rem 1.5rem 0;
      display: flex;
      gap: 0.5rem;
      overflow-x: auto;
    }}

    .tab-btn {{
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-family: var(--font-family);
      font-size: 0.9rem;
      font-weight: 600;
      padding: 0.75rem 1rem;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: all var(--transition-fast);
      white-space: nowrap;
    }}
    .tab-btn:hover {{
      color: var(--text-main);
    }}
    .tab-btn.active {{
      color: var(--text-main);
      border-bottom-color: var(--text-main);
    }}

    /* ==========================================================================
       4. 页面主体容器与选项卡视窗
       ========================================================================== */
    .main-container {{
      flex: 1;
      max-width: 1440px;
      width: 100%;
      margin: 0 auto;
      padding: 1.5rem;
    }}

    .tab-pane {{
      display: none;
      animation: fadeIn 0.25s ease;
    }}
    .tab-pane.active {{
      display: block;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(4px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* 通用卡片容器 */
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-lg);
      padding: 1.25rem;
      margin-bottom: 1.5rem;
      box-shadow: var(--shadow-sm);
      transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
    }}
    .card:hover {{
      border-color: var(--border-subtle);
    }}

    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--border-main);
    }}
    .card-title {{
      font-size: 1.05rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .card-subtitle {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }}

    /* 按钮规范 */
    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      font-family: var(--font-family);
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.5rem 1rem;
      border-radius: var(--radius-md);
      cursor: pointer;
      border: 1px solid transparent;
      transition: all var(--transition-fast);
    }}
    .btn-primary {{
      background: var(--accent-primary);
      color: var(--accent-primary-fg);
      border-color: var(--accent-primary);
    }}
    .btn-primary:hover {{
      background: var(--accent-primary-hover);
      border-color: var(--accent-primary-hover);
    }}
    .btn-secondary {{
      background: var(--bg-card-subtle);
      color: var(--text-main);
      border-color: var(--border-main);
    }}
    .btn-secondary:hover {{
      background: var(--bg-card-hover);
      border-color: var(--border-subtle);
    }}
    .btn-sm {{
      padding: 0.3rem 0.65rem;
      font-size: 0.78rem;
      border-radius: var(--radius-sm);
    }}
    .btn-icon {{
      padding: 0.4rem;
      border-radius: var(--radius-sm);
    }}

    /* ==========================================================================
       5. 模块 1：⚡ 模型对决与在线打靶 (Arena)
       ========================================================================== */
    .input-section {{
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}

    .textarea-wrapper {{
      position: relative;
    }}

    .arena-textarea {{
      width: 100%;
      min-height: 84px;
      max-height: 200px;
      padding: 0.85rem 1rem;
      background: var(--bg-input);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      color: var(--text-main);
      font-family: var(--font-family);
      font-size: 0.95rem;
      resize: vertical;
      line-height: 1.6;
      transition: border-color var(--transition-fast);
    }}
    .arena-textarea:focus {{
      outline: none;
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px rgba(128, 128, 128, 0.2);
    }}

    .char-count {{
      position: absolute;
      right: 12px;
      bottom: 10px;
      font-size: 0.75rem;
      color: var(--text-muted);
      background: var(--bg-card);
      padding: 2px 6px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-main);
      pointer-events: none;
    }}

    .arena-controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}

    .presets-bar {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }}
    .presets-label {{
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 0.3rem;
    }}
    .preset-pill {{
      font-size: 0.75rem;
      padding: 0.25rem 0.6rem;
      border-radius: var(--radius-full);
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all var(--transition-fast);
      white-space: nowrap;
    }}
    .preset-pill:hover {{
      color: var(--text-main);
      background: var(--bg-card-hover);
      border-color: var(--border-subtle);
    }}
    .preset-pill.active {{
      background: var(--text-main);
      border-color: var(--text-main);
      color: var(--bg-body);
      font-weight: 600;
    }}

    /* 用户自定义用例抽屉 */
    .custom-cases-panel {{
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      padding: 1rem;
      margin-top: 1rem;
    }}
    .custom-cases-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.75rem;
    }}
    .custom-cases-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 0.75rem;
      max-height: 260px;
      overflow-y: auto;
      padding-right: 4px;
    }}
    .custom-case-item {{
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      padding: 0.75rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 0.5rem;
      transition: all var(--transition-fast);
    }}
    .custom-case-item:hover {{
      border-color: var(--border-subtle);
    }}
    .custom-case-meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }}
    .custom-case-tag {{
      font-size: 0.7rem;
      padding: 0.1rem 0.4rem;
      border-radius: var(--radius-sm);
      background: var(--bg-card-hover);
      border: 1px solid var(--border-subtle);
      color: var(--text-secondary);
      font-weight: 600;
    }}
    .custom-case-time {{
      font-size: 0.7rem;
      color: var(--text-muted);
    }}
    .custom-case-text {{
      font-size: 0.8rem;
      color: var(--text-main);
      line-height: 1.4;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .custom-case-note {{
      font-size: 0.75rem;
      color: var(--text-muted);
      font-style: italic;
    }}
    .custom-case-actions {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 0.4rem;
    }}

    /* 差异分析报告视窗 (Diff Mode) - 极简黑白灰 */
    .diff-alert-card {{
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
      border-left: 4px solid var(--text-main);
      border-radius: var(--radius-lg);
      padding: 1rem 1.25rem;
      margin-bottom: 1.5rem;
    }}
    .diff-alert-card.identical {{
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
      border-left: 4px solid var(--border-subtle);
    }}
    .diff-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }}
    .diff-title {{
      font-size: 0.95rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .diff-desc {{
      font-size: 0.85rem;
      color: var(--text-secondary);
      line-height: 1.5;
    }}
    .diff-details-list {{
      margin-top: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}
    .diff-detail-item {{
      font-size: 0.82rem;
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      padding: 0.5rem 0.75rem;
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
    }}

    /* 双模型同屏对比区 */
    .arena-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    @media (max-width: 900px) {{
      .arena-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    .model-col {{
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-lg);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      box-shadow: var(--shadow-sm);
    }}
    .model-col.macbert {{
      border-top: 3px solid var(--accent-macbert);
    }}
    .model-col.bert {{
      border-top: 3px solid var(--accent-bert);
    }}

    .model-meta-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--border-main);
    }}
    .model-tag-name {{
      font-size: 0.95rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .model-stats-badges {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .stat-chip {{
      font-size: 0.75rem;
      padding: 0.2rem 0.5rem;
      border-radius: var(--radius-sm);
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
      color: var(--text-secondary);
      font-weight: 500;
    }}

    /* 实体高亮文本视窗 */
    .visual-text-box {{
      background: var(--bg-input);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      padding: 1rem;
      font-size: 1rem;
      line-height: 2;
      min-height: 100px;
      word-break: break-all;
    }}

    /* 实体标签高亮样式与 Tooltip (极简黑白灰高对比度体系) */
    .entity-span {{
      display: inline-flex;
      align-items: center;
      gap: 0.2rem;
      padding: 0.12rem 0.45rem;
      margin: 0 0.15rem;
      border-radius: var(--radius-sm);
      cursor: pointer;
      position: relative;
      font-weight: 600;
      transition: all var(--transition-fast);
      background: var(--badge-bg, var(--bg-card-subtle));
      color: var(--badge-color, var(--text-main));
      border: 1px solid var(--badge-border, var(--border-subtle));
    }}
    .entity-span:hover {{
      transform: translateY(-1px);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }}

    .ent-label {{
      font-size: 0.68rem;
      opacity: 0.8;
      font-weight: 700;
      margin-left: 0.2rem;
    }}

    .ent-tooltip {{
      visibility: hidden;
      opacity: 0;
      position: absolute;
      bottom: calc(100% + 6px);
      left: 50%;
      transform: translateX(-50%);
      background: #000000;
      color: #ffffff;
      padding: 0.4rem 0.7rem;
      border-radius: var(--radius-sm);
      font-size: 0.75rem;
      font-family: var(--font-family);
      font-weight: 400;
      line-height: 1.4;
      white-space: nowrap;
      z-index: 50;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5);
      border: 1px solid #333333;
      pointer-events: none;
      transition: opacity var(--transition-fast), visibility var(--transition-fast);
    }}
    .entity-span:hover .ent-tooltip {{
      visibility: visible;
      opacity: 1;
    }}
    [data-theme="light"] .ent-tooltip {{
      background: #000000;
      color: #ffffff;
      border: 1px solid #000000;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }}

    /* 10 类别高亮色彩配置 (兼容纯色变量) */
    .badge-address {{ --badge-color: #ffffff; }}
    .badge-book {{ --badge-color: #e5e5e5; }}
    .badge-company {{ --badge-color: #ffffff; }}
    .badge-game {{ --badge-color: #d4d4d4; }}
    .badge-government {{ --badge-color: #ffffff; }}
    .badge-movie {{ --badge-color: #b0b0b0; }}
    .badge-name {{ --badge-color: #000000; }}
    .badge-organization {{ --badge-color: #ffffff; }}
    .badge-position {{ --badge-color: #8c8c8c; }}
    .badge-scene {{ --badge-color: #f0f0f0; }}

    [data-theme="light"] .badge-address {{ --badge-color: #111111; }}
    [data-theme="light"] .badge-book {{ --badge-color: #222222; }}
    [data-theme="light"] .badge-company {{ --badge-color: #000000; }}
    [data-theme="light"] .badge-game {{ --badge-color: #333333; }}
    [data-theme="light"] .badge-government {{ --badge-color: #000000; }}
    [data-theme="light"] .badge-movie {{ --badge-color: #555555; }}
    [data-theme="light"] .badge-name {{ --badge-color: #ffffff; }}
    [data-theme="light"] .badge-organization {{ --badge-color: #000000; }}
    [data-theme="light"] .badge-position {{ --badge-color: #555555; }}
    [data-theme="light"] .badge-scene {{ --badge-color: #111111; }}

    /* 结构化实体列表 */
    .entities-chips-container {{
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      max-height: 220px;
      overflow-y: auto;
      padding-right: 4px;
    }}
    .entity-chip-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      padding: 0.4rem 0.6rem;
      font-size: 0.8rem;
    }}
    .entity-chip-left {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .entity-chip-span {{
      font-size: 0.7rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}
    .entity-chip-right {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-shrink: 0;
    }}
    .conf-bar-bg {{
      width: 50px;
      height: 6px;
      background: var(--border-main);
      border-radius: var(--radius-full);
      overflow: hidden;
    }}
    .conf-bar-fill {{
      height: 100%;
      background: var(--text-main);
      border-radius: var(--radius-full);
    }}
    .conf-percent {{
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
      width: 36px;
      text-align: right;
    }}

    /* 底层 Token 探针 */
    .probe-toggle {{
      background: var(--bg-card-subtle);
      border: 1px dashed var(--border-main);
      border-radius: var(--radius-md);
      padding: 0.6rem 1rem;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      color: var(--text-secondary);
      font-size: 0.85rem;
      font-weight: 600;
      transition: all var(--transition-fast);
    }}
    .probe-toggle:hover {{
      background: var(--bg-card-hover);
      color: var(--text-main);
      border-color: var(--border-subtle);
    }}

    .probe-table-wrapper {{
      display: none;
      margin-top: 1rem;
      overflow-x: auto;
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
    }}
    .probe-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
      text-align: left;
    }}
    .probe-table th, .probe-table td {{
      padding: 0.5rem 0.75rem;
      border-bottom: 1px solid var(--border-main);
      white-space: nowrap;
    }}
    .probe-table th {{
      background: var(--bg-card-subtle);
      color: var(--text-secondary);
      font-weight: 600;
    }}
    .probe-table tr:hover td {{
      background: var(--bg-card-hover);
    }}
    .probe-table tr.row-diff td {{
      background: var(--bg-card-hover) !important;
      font-weight: 600;
    }}
    .bio-tag {{
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      padding: 0.1rem 0.35rem;
      border-radius: var(--radius-sm);
      font-weight: 600;
    }}
    .bio-tag.bio-O {{
      color: var(--text-muted);
      background: var(--bg-card-subtle);
    }}
    .bio-tag.bio-B {{
      background: var(--bg-card-hover);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
    }}
    .bio-tag.bio-I {{
      background: var(--bg-card-subtle);
      color: var(--text-secondary);
      border: 1px dashed var(--border-main);
    }}

    /* ==========================================================================
       6. 模块 2：💻 复现代码深度透视 (Code Walkthrough)
       ========================================================================== */
    .code-tabs-bar {{
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
      border-bottom: 1px solid var(--border-main);
      padding-bottom: 0.5rem;
    }}
    .code-tab-btn {{
      padding: 0.4rem 0.8rem;
      font-size: 0.85rem;
      font-family: var(--font-mono);
      border-radius: var(--radius-md);
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all var(--transition-fast);
    }}
    .code-tab-btn:hover {{
      color: var(--text-main);
    }}
    .code-tab-btn.active {{
      background: var(--text-main);
      color: var(--bg-body);
      border-color: var(--text-main);
      font-weight: 600;
    }}

    .code-walkthrough-layout {{
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 1.5rem;
      align-items: start;
    }}
    @media (max-width: 1024px) {{
      .code-walkthrough-layout {{
        grid-template-columns: 1fr;
      }}
    }}

    .code-annotations-col {{
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .annotation-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      padding: 1rem;
      border-left: 4px solid var(--text-main);
    }}
    .annotation-title {{
      font-size: 0.9rem;
      font-weight: 700;
      margin-bottom: 0.4rem;
      color: var(--text-main);
    }}
    .annotation-snippet {{
      background: var(--bg-input);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-sm);
      padding: 0.4rem 0.6rem;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      margin-bottom: 0.5rem;
      color: var(--text-main);
      white-space: pre-wrap;
    }}
    .annotation-text {{
      font-size: 0.82rem;
      color: var(--text-secondary);
      line-height: 1.5;
    }}

    .code-viewer-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      overflow: hidden;
    }}
    .code-viewer-header {{
      background: var(--bg-card-subtle);
      padding: 0.6rem 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-main);
    }}
    .code-filename {{
      font-family: var(--font-mono);
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-main);
    }}
    .code-block-container {{
      max-height: 640px;
      overflow: auto;
      background: #0d0d0d;
      color: #f5f5f5;
      font-family: var(--font-mono);
      font-size: 0.82rem;
      line-height: 1.6;
      padding: 1rem;
      border: 1px solid var(--border-main);
    }}
    [data-theme="light"] .code-block-container {{
      background: #f8f8f8;
      color: #171717;
    }}
    .code-block-container pre {{
      margin: 0;
    }}

    /* ==========================================================================
       7. 模块 3：📊 实测学术指标大盘 (Benchmark Dashboard)
       ========================================================================== */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}
    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-lg);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}
    .kpi-card.highlight {{
      background: var(--bg-card);
      border: 2px solid var(--text-main);
    }}
    .kpi-title {{
      font-size: 0.8rem;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .kpi-value-row {{
      display: flex;
      align-items: baseline;
      gap: 0.5rem;
    }}
    .kpi-value {{
      font-size: 1.8rem;
      font-weight: 800;
      color: var(--text-main);
      font-family: var(--font-mono);
    }}
    .kpi-delta {{
      font-size: 0.85rem;
      font-weight: 700;
      padding: 0.1rem 0.4rem;
      border-radius: var(--radius-sm);
    }}
    .kpi-delta.pos {{
      background: var(--bg-card-hover);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
    }}
    .kpi-subtext {{
      font-size: 0.75rem;
      color: var(--text-secondary);
    }}

    .chart-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-lg);
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .chart-controls {{
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1.25rem;
    }}
    .chart-btn {{
      padding: 0.35rem 0.75rem;
      font-size: 0.8rem;
      border-radius: var(--radius-sm);
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-main);
      color: var(--text-secondary);
      cursor: pointer;
    }}
    .chart-btn.active {{
      background: var(--text-main);
      color: var(--bg-body);
      border-color: var(--text-main);
      font-weight: 600;
    }}

    .svg-chart-wrapper {{
      width: 100%;
      overflow-x: auto;
    }}

    .benchmark-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin-top: 1rem;
    }}
    .benchmark-table th, .benchmark-table td {{
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid var(--border-main);
      text-align: left;
    }}
    .benchmark-table th {{
      background: var(--bg-card-subtle);
      color: var(--text-secondary);
      font-weight: 600;
    }}
    .benchmark-table tr:hover td {{
      background: var(--bg-card-hover);
    }}

    /* ==========================================================================
       8. 模块 4：🔍 真实错例深度展厅 (Bad Case Gallery)
       ========================================================================== */
    .gallery-filter-bar {{
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1.25rem;
      flex-wrap: wrap;
    }}
    .gallery-filter-btn {{
      padding: 0.4rem 0.9rem;
      font-size: 0.85rem;
      border-radius: var(--radius-full);
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all var(--transition-fast);
    }}
    .gallery-filter-btn.active {{
      background: var(--text-main);
      color: var(--bg-body);
      border-color: var(--text-main);
      font-weight: 600;
    }}

    .cases-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
      gap: 1.25rem;
    }}
    @media (max-width: 640px) {{
      .cases-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    .case-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-lg);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
    }}
    .case-card:hover {{
      border-color: var(--border-subtle);
      box-shadow: var(--shadow-md);
    }}
    .case-top-meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }}
    .case-id-badge {{
      font-family: var(--font-mono);
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.15rem 0.5rem;
      border-radius: var(--radius-sm);
      background: var(--bg-card-subtle);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
    }}
    .case-title {{
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--text-main);
      margin-top: 0.2rem;
    }}
    .case-text-box {{
      background: var(--bg-input);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      padding: 0.75rem;
      font-size: 0.85rem;
      line-height: 1.5;
    }}
    .comparison-box {{
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      background: var(--bg-card-subtle);
      border-radius: var(--radius-md);
      padding: 0.75rem;
      font-size: 0.8rem;
    }}
    .comp-row {{
      display: flex;
      align-items: flex-start;
      gap: 0.5rem;
    }}
    .comp-label {{
      font-weight: 600;
      width: 72px;
      flex-shrink: 0;
      font-size: 0.75rem;
    }}
    .comp-entities {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.3rem;
    }}
    .case-reason {{
      font-size: 0.8rem;
      color: var(--text-secondary);
      line-height: 1.5;
    }}
    .case-card-footer {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      padding-top: 0.5rem;
      border-top: 1px solid var(--border-main);
    }}

    /* ==========================================================================
       9. 弹窗与吐司通知 (Modal & Toast)
       ========================================================================== */
    .modal-overlay {{
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(4px);
      z-index: 200;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }}
    .modal-overlay.active {{
      display: flex;
    }}
    .modal-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-lg);
      width: 100%;
      max-width: 520px;
      padding: 1.5rem;
      box-shadow: var(--shadow-lg);
      animation: modalSlide 0.2s ease;
    }}
    @keyframes modalSlide {{
      from {{ transform: scale(0.95); opacity: 0; }}
      to {{ transform: scale(1); opacity: 1; }}
    }}
    .modal-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }}
    .modal-title {{
      font-size: 1.05rem;
      font-weight: 700;
    }}
    .form-group {{
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      margin-bottom: 1rem;
    }}
    .form-label {{
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-secondary);
    }}
    .form-input, .form-textarea {{
      background: var(--bg-input);
      border: 1px solid var(--border-main);
      border-radius: var(--radius-md);
      color: var(--text-main);
      padding: 0.5rem 0.75rem;
      font-family: var(--font-family);
      font-size: 0.85rem;
    }}
    .form-input:focus, .form-textarea:focus {{
      outline: none;
      border-color: var(--border-focus);
    }}
    .modal-actions {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 0.5rem;
      margin-top: 1.25rem;
    }}

    .toast-container {{
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 300;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      pointer-events: none;
    }}
    .toast {{
      background: var(--text-main);
      color: var(--bg-body);
      padding: 0.6rem 1.1rem;
      border-radius: var(--radius-md);
      font-size: 0.85rem;
      font-weight: 600;
      box-shadow: var(--shadow-lg);
      border: 1px solid var(--border-main);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      animation: toastIn 0.25s ease;
      pointer-events: auto;
    }}
    @keyframes toastIn {{
      from {{ transform: translateY(10px); opacity: 0; }}
      to {{ transform: translateY(0); opacity: 1; }}
    }}

    /* 页脚 */
    .footer {{
      margin-top: auto;
      border-top: 1px solid var(--border-main);
      padding: 1.25rem 1.5rem;
      text-align: center;
      font-size: 0.8rem;
      color: var(--text-muted);
      background: var(--bg-card);
    }}
  </style>
</head>
<body>

  <!-- 1. 顶部导航与状态条 -->
  <header class="navbar">
    <div class="nav-brand">
      <div class="nav-logo">M</div>
      <div class="nav-title-group">
        <h1>MacBERT × CLUENER2020 <span class="nav-badge">v2.0 Academic Workbench</span></h1>
        <div class="sub-title">细粒度中文命名实体识别复现工作台与全栈模型对决体验系统</div>
      </div>
    </div>
    <div class="nav-actions">
      <div class="status-badge" id="backendStatusBadge">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusText">正在探测后端环境...</span>
      </div>
      <button class="theme-toggle-btn" id="themeToggleBtn" title="切换深色/浅色主题">
        <svg id="themeIconSun" style="display:none" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
        <svg id="themeIconMoon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
      </button>
    </div>
  </header>

  <!-- 2. 选项卡导航栏 -->
  <nav class="tab-nav-container">
    <button class="tab-btn active" data-tab="tab-arena">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
      ⚡ 模型对决与在线打靶 (Arena)
    </button>
    <button class="tab-btn" data-tab="tab-code">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>
      💻 复现代码深度透视 (Codebase)
    </button>
    <button class="tab-btn" data-tab="tab-benchmark">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
      📊 实测学术指标大盘 (Benchmark)
    </button>
    <button class="tab-btn" data-tab="tab-gallery">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
      🔍 真实错例深度展厅 (Bad Cases)
    </button>
  </nav>

  <!-- 3. 主界面容器 -->
  <main class="main-container">

    <!-- ====================================================================
         TAB 1: 模型对决与在线打靶 (Arena)
         ==================================================================== -->
    <section class="tab-pane active" id="tab-arena">
      
      <!-- 离线模式降级提醒横幅（默认隐藏，离线时展现） -->
      <div id="offlineNoticeBanner" style="display:none; background: var(--bg-card-subtle); border: 1px solid var(--border-main); border-left: 4px solid var(--text-muted); border-radius: var(--radius-md); padding: 0.75rem 1rem; margin-bottom: 1.25rem; font-size: 0.85rem; color: var(--text-main); display: flex; align-items: center; justify-content: space-between; gap: 1rem;">
        <div>
          <strong>当前处于离线演示模式 (Offline Simulation Active)</strong>：
          未检测到后端 Flask 服务，系统已自动切换至内置的 100% 离线推演引擎与真实预置预测集。
          若需体验 PyTorch 真实权重实时推理，只需在终端执行 <code>python app.py</code>。
        </div>
      </div>

      <!-- 输入与打靶控制卡片 -->
      <div class="card">
        <div class="input-section">
          <div class="textarea-wrapper">
            <textarea id="arenaInput" class="arena-textarea" placeholder="请输入任意中文文本进行命名实体识别打靶测试（支持地址、书籍、公司、游戏、政府、电影、姓名、组织、职位、景点 10 类实体抽取）..." maxlength="128"></textarea>
            <span class="char-count" id="charCount">0 / 128 字</span>
          </div>

          <div class="arena-controls">
            <div class="presets-bar">
              <span class="presets-label">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                预设经典例:
              </span>
              <div id="presetPillsContainer" style="display:inline-flex; gap:0.4rem; flex-wrap:wrap;"></div>
            </div>

            <div style="display:flex; align-items:center; gap:0.5rem;">
              <button class="btn btn-secondary btn-sm" id="btnToggleCustomCases">
                📁 自定义用例集 (<span id="customCasesBadge">0</span>)
              </button>
              <button class="btn btn-secondary btn-sm" id="btnOpenSaveModal">
                ⭐ 保存当前用例
              </button>
              <button class="btn btn-secondary btn-sm" id="btnClearInput">
                🧹 清空
              </button>
              <button class="btn btn-primary" id="btnRunArena">
                <span id="btnRunIcon">⚡</span>
                <span id="btnRunText">开始模型对决分析</span>
              </button>
            </div>
          </div>
        </div>

        <!-- 自定义用例折叠面板 -->
        <div class="custom-cases-panel" id="customCasesPanel" style="display:none;">
          <div class="custom-cases-header">
            <div style="font-size:0.9rem; font-weight:700; display:flex; align-items:center; gap:0.4rem;">
              <span>📁 我的自定义测试用例</span>
              <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">(基于 LocalStorage 本地持久化)</span>
            </div>
            <div style="display:flex; gap:0.4rem;">
              <button class="btn btn-secondary btn-sm" id="btnExportCases">📥 导出 JSON</button>
              <button class="btn btn-secondary btn-sm" id="btnImportCases">📤 导入 JSON</button>
              <input type="file" id="importFileInput" accept=".json" style="display:none">
            </div>
          </div>
          <div class="custom-cases-grid" id="customCasesGrid"></div>
        </div>
      </div>

      <!-- 差异对比视窗 (Diff Mode Alert) -->
      <div id="diffAlertContainer"></div>

      <!-- 双模型同屏对决展示网格 (MacBERT vs BERT) -->
      <div class="arena-grid">
        <!-- MacBERT 列 -->
        <div class="model-col macbert">
          <div class="model-meta-bar">
            <div class="model-tag-name" style="color:var(--accent-macbert);">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
              Chinese-MacBERT-base (Proposed)
            </div>
            <div class="model-stats-badges">
              <span class="stat-chip" id="macbertLatency">耗时: -- ms</span>
              <span class="stat-chip" id="macbertCount">实体: --</span>
            </div>
          </div>
          <div class="visual-text-box" id="macbertVisualBox">等待输入后执行推理...</div>
          <div style="font-size:0.8rem; font-weight:700; color:var(--text-secondary); margin-top:0.25rem;">识别实体明细:</div>
          <div class="entities-chips-container" id="macbertEntitiesList">
            <div style="font-size:0.8rem; color:var(--text-muted); padding:0.5rem 0;">暂无实体</div>
          </div>
        </div>

        <!-- BERT 列 -->
        <div class="model-col bert">
          <div class="model-meta-bar">
            <div class="model-tag-name" style="color:var(--text-secondary);">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
              BERT-base-Chinese (Baseline)
            </div>
            <div class="model-stats-badges">
              <span class="stat-chip" id="bertLatency">耗时: -- ms</span>
              <span class="stat-chip" id="bertCount">实体: --</span>
            </div>
          </div>
          <div class="visual-text-box" id="bertVisualBox">等待输入后执行推理...</div>
          <div style="font-size:0.8rem; font-weight:700; color:var(--text-secondary); margin-top:0.25rem;">识别实体明细:</div>
          <div class="entities-chips-container" id="bertEntitiesList">
            <div style="font-size:0.8rem; color:var(--text-muted); padding:0.5rem 0;">暂无实体</div>
          </div>
        </div>
      </div>

      <!-- 底层 Token 探针 (可展开) -->
      <div class="card" style="padding:1rem;">
        <button class="probe-toggle" id="btnToggleProbe">
          <span style="display:flex; align-items:center; gap:0.5rem;">
            <span>🔬 底层 Token 探针 (逐字预测 BIO 标签与模型置信度)</span>
            <span class="nav-badge" id="probeDiffCountBadge" style="display:none;">发现分歧</span>
          </span>
          <span id="probeToggleArrow">▼ 展开探针</span>
        </button>
        <div class="probe-table-wrapper" id="probeTableWrapper">
          <table class="probe-table">
            <thead>
              <tr>
                <th style="width:48px;">索引</th>
                <th style="width:60px;">字符</th>
                <th>BERT 预测 BIO</th>
                <th>BERT 置信度</th>
                <th>MacBERT 预测 BIO</th>
                <th>MacBERT 置信度</th>
                <th>一致性</th>
              </tr>
            </thead>
            <tbody id="probeTableBody"></tbody>
          </table>
        </div>
      </div>

    </section>

    <!-- ====================================================================
         TAB 2: 复现代码深度透视 (Codebase Walkthrough)
         ==================================================================== -->
    <section class="tab-pane" id="tab-code">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">💻 核心复现代码与算法机制深度透视</div>
            <div class="card-subtitle">系统解析 CLUENER 数据流水线字符级对齐、BERT/MacBERT 训练微调架构及 Seqeval 严格实体匹配评测算法</div>
          </div>
        </div>

        <div class="code-tabs-bar" id="codeFilesNav"></div>

        <div class="code-walkthrough-layout">
          <!-- 左侧：技术考点与原理联动批注 -->
          <div class="code-annotations-col" id="codeAnnotationsContainer"></div>

          <!-- 右侧：完整代码查看器 -->
          <div class="code-viewer-card">
            <div class="code-viewer-header">
              <span class="code-filename" id="currentCodeFilename">dataset.py</span>
              <button class="btn btn-secondary btn-sm" id="btnCopyCode">📋 复制代码</button>
            </div>
            <div class="code-block-container">
              <pre><code id="codeSourceDisplay"># 正在载入源码...</code></pre>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ====================================================================
         TAB 3: 实测学术指标大盘 (Benchmark Dashboard)
         ==================================================================== -->
    <section class="tab-pane" id="tab-benchmark">
      <!-- 顶部核心 KPI 卡片 -->
      <div class="kpi-grid">
        <div class="kpi-card highlight">
          <div class="kpi-title">MacBERT 评测综合 F1</div>
          <div class="kpi-value-row">
            <span class="kpi-value" id="kpiMacbertF1">76.58%</span>
            <span class="kpi-delta pos">+1.62%</span>
          </div>
          <div class="kpi-subtext">较原生 BERT (74.96%) 取得显著突破提升</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">原生 BERT 评测 F1</div>
          <div class="kpi-value-row">
            <span class="kpi-value" id="kpiBertF1">74.96%</span>
            <span class="kpi-subtext">(Dev: 76.18%)</span>
          </div>
          <div class="kpi-subtext">微调基线收敛表现</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">验证集最低损失 (Loss)</div>
          <div class="kpi-value-row">
            <span class="kpi-value">0.1935</span>
            <span class="kpi-delta pos">-5.6%</span>
          </div>
          <div class="kpi-subtext">MacBERT 收敛拟合更优 (BERT 0.2050)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">最优收敛轮次 (Best Epoch)</div>
          <div class="kpi-value-row">
            <span class="kpi-value">Epoch 2</span>
            <span class="kpi-subtext">早于 BERT (Ep.3)</span>
          </div>
          <div class="kpi-subtext">全词掩码增强先验，收敛速度更快</div>
        </div>
      </div>

      <!-- 10 类实体细粒度指标对比柱状图 -->
      <div class="chart-card">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem; margin-bottom:1rem;">
          <div>
            <div style="font-size:1.05rem; font-weight:700;">10 类细粒度实体性能对比 (Fine-Grained Category Breakdown)</div>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.2rem;">基于 Seqeval 严格实体匹配标准 (Exact Span + Category Match)</div>
          </div>
          <div class="chart-controls">
            <button class="chart-btn active" data-metric="f1">F1-Score</button>
            <button class="chart-btn" data-metric="precision">Precision (精确率)</button>
            <button class="chart-btn" data-metric="recall">Recall (召回率)</button>
          </div>
        </div>
        <div class="svg-chart-wrapper" id="svgChartContainer"></div>
      </div>

      <!-- 10 类详细对比指标表格 -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">📋 细粒度实体分类指标明细大表</div>
        </div>
        <div style="overflow-x:auto;">
          <table class="benchmark-table">
            <thead>
              <tr>
                <th>实体类别</th>
                <th>实体数量 (Support)</th>
                <th>BERT 精确率 / 召回 / F1</th>
                <th>MacBERT 精确率 / 召回 / F1</th>
                <th>F1 差异增益 (Delta)</th>
                <th>模型优势评判</th>
              </tr>
            </thead>
            <tbody id="benchmarkTableBody"></tbody>
          </table>
        </div>
      </div>

      <!-- 实验超参数与基准规范 -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">⚙️ 实验复现环境与训练超参数基准表</div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:1rem; font-size:0.85rem;">
          <div style="background:var(--bg-card-subtle); padding:0.75rem 1rem; border-radius:var(--radius-md);">
            <strong>预训练基础模型：</strong> <code>bert-base-chinese</code> vs <code>hfl/chinese-macbert-base</code>
          </div>
          <div style="background:var(--bg-card-subtle); padding:0.75rem 1rem; border-radius:var(--radius-md);">
            <strong>隐藏层维度 / 注意力头：</strong> 768 维 / 12 Heads / 12 Layers
          </div>
          <div style="background:var(--bg-card-subtle); padding:0.75rem 1rem; border-radius:var(--radius-md);">
            <strong>微调批大小 (Batch Size)：</strong> 16
          </div>
          <div style="background:var(--bg-card-subtle); padding:0.75rem 1rem; border-radius:var(--radius-md);">
            <strong>最大学习率 (Learning Rate)：</strong> 3e-5 (配合 AdamW 与 0.1 Warmup)
          </div>
          <div style="background:var(--bg-card-subtle); padding:0.75rem 1rem; border-radius:var(--radius-md);">
            <strong>序列最大长度 (Max Length)：</strong> 128 Token
          </div>
          <div style="background:var(--bg-card-subtle); padding:0.75rem 1rem; border-radius:var(--radius-md);">
            <strong>评测样本规模：</strong> 训练集 10,748 句 / 开发集 1,343 句 (3,072 实体)
          </div>
        </div>
      </div>
    </section>

    <!-- ====================================================================
         TAB 4: 真实错例深度展厅 (Bad Case Gallery)
         ==================================================================== -->
    <section class="tab-pane" id="tab-gallery">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">🔍 真实错例与典型案例全集展厅 (Bad Case Gallery)</div>
            <div class="card-subtitle">系统挖掘 1,343 条真实样本中 MacBERT 对 BERT 的边界纠错修复，以及中文命名实体识别的固有边界瓶颈</div>
          </div>
        </div>

        <!-- 错例分类筛选 -->
        <div class="gallery-filter-bar">
          <button class="gallery-filter-btn active" data-filter="all">全部代表性案例 (14)</button>
          <button class="gallery-filter-btn" data-filter="advantages">类别 A：MacBERT 优势/纠错案例 (8)</button>
          <button class="gallery-filter-btn" data-filter="hard">类别 B：中文 NER 固有瓶颈与双错案例 (6)</button>
        </div>

        <!-- 错例网格卡片 -->
        <div class="cases-grid" id="casesGridContainer"></div>
      </div>
    </section>

  </main>

  <!-- 4. 自定义用例保存/添加弹窗 (Modal) -->
  <div class="modal-overlay" id="saveCaseModal">
    <div class="modal-card">
      <div class="modal-header">
        <div class="modal-title" id="modalCaseTitleHeader">⭐ 保存为自定义测试用例</div>
        <button class="btn btn-secondary btn-icon" id="btnCloseModal">✕</button>
      </div>
      <div class="form-group">
        <label class="form-label">用例标题 / 场景名称 *</label>
        <input type="text" id="modalCaseTitle" class="form-input" placeholder="例如：长金融机构分支机构识别">
      </div>
      <div class="form-group">
        <label class="form-label">标签分类 (Tag)</label>
        <input type="text" id="modalCaseTag" class="form-input" placeholder="例如：公司/职位/长实体">
      </div>
      <div class="form-group">
        <label class="form-label">待测试中文文本 *</label>
        <textarea id="modalCaseText" class="form-textarea" rows="3"></textarea>
      </div>
      <div class="form-group">
        <label class="form-label">测试备注与预期说明 (可选)</label>
        <input type="text" id="modalCaseNote" class="form-input" placeholder="例如：预期 MacBERT 能避免将分支机构切碎">
      </div>
      <div class="modal-actions">
        <button class="btn btn-secondary" id="btnCancelSave">取消</button>
        <button class="btn btn-primary" id="btnConfirmSave">确认保存</button>
      </div>
    </div>
  </div>

  <!-- 5. 吐司消息通知容器 -->
  <div class="toast-container" id="toastContainer"></div>

  <!-- 6. 统一页脚 -->
  <footer class="footer">
    MacBERT × CLUENER2020 细粒度中文命名实体识别复现工作台 · Reproduction Workbench & Live Model Arena · 基于 PyTorch 与 Transformers
  </footer>

  <!-- ==========================================================================
       10. 离线平滑降级静态数据包注入 (Static Offline Bundle)
       ========================================================================== -->
  <script>
    window.__NER_BUNDLE__ = {bundle_json_str};
  </script>

  <!-- ==========================================================================
       11. 核心前端交互与推演逻辑脚本
       ========================================================================== -->
  <script>
    (function() {{
      'use strict';

      // 全局状态管理
      const state = {{
        isOnline: false,
        activeTab: 'tab-arena',
        activeMetric: 'f1',
        activeFilter: 'all',
        activeCodeFile: 'dataset',
        customCases: [],
        editingCaseId: null,
        lastPredictResult: null
      }};

      const bundle = window.__NER_BUNDLE__ || {{}};

      // DOM 元素引用
      const el = {{
        backendStatusBadge: document.getElementById('backendStatusBadge'),
        statusDot: document.getElementById('statusDot'),
        statusText: document.getElementById('statusText'),
        offlineNoticeBanner: document.getElementById('offlineNoticeBanner'),
        themeToggleBtn: document.getElementById('themeToggleBtn'),
        themeIconSun: document.getElementById('themeIconSun'),
        themeIconMoon: document.getElementById('themeIconMoon'),
        
        // Arena
        arenaInput: document.getElementById('arenaInput'),
        charCount: document.getElementById('charCount'),
        presetPillsContainer: document.getElementById('presetPillsContainer'),
        btnRunArena: document.getElementById('btnRunArena'),
        btnRunIcon: document.getElementById('btnRunIcon'),
        btnRunText: document.getElementById('btnRunText'),
        btnClearInput: document.getElementById('btnClearInput'),
        btnOpenSaveModal: document.getElementById('btnOpenSaveModal'),
        btnToggleCustomCases: document.getElementById('btnToggleCustomCases'),
        customCasesBadge: document.getElementById('customCasesBadge'),
        customCasesPanel: document.getElementById('customCasesPanel'),
        customCasesGrid: document.getElementById('customCasesGrid'),
        btnExportCases: document.getElementById('btnExportCases'),
        btnImportCases: document.getElementById('btnImportCases'),
        importFileInput: document.getElementById('importFileInput'),
        
        diffAlertContainer: document.getElementById('diffAlertContainer'),
        macbertLatency: document.getElementById('macbertLatency'),
        macbertCount: document.getElementById('macbertCount'),
        macbertVisualBox: document.getElementById('macbertVisualBox'),
        macbertEntitiesList: document.getElementById('macbertEntitiesList'),
        
        bertLatency: document.getElementById('bertLatency'),
        bertCount: document.getElementById('bertCount'),
        bertVisualBox: document.getElementById('bertVisualBox'),
        bertEntitiesList: document.getElementById('bertEntitiesList'),
        
        btnToggleProbe: document.getElementById('btnToggleProbe'),
        probeToggleArrow: document.getElementById('probeToggleArrow'),
        probeTableWrapper: document.getElementById('probeTableWrapper'),
        probeTableBody: document.getElementById('probeTableBody'),
        probeDiffCountBadge: document.getElementById('probeDiffCountBadge'),
        
        // Code
        codeFilesNav: document.getElementById('codeFilesNav'),
        codeAnnotationsContainer: document.getElementById('codeAnnotationsContainer'),
        currentCodeFilename: document.getElementById('currentCodeFilename'),
        codeSourceDisplay: document.getElementById('codeSourceDisplay'),
        btnCopyCode: document.getElementById('btnCopyCode'),
        
        // Benchmark
        kpiMacbertF1: document.getElementById('kpiMacbertF1'),
        kpiBertF1: document.getElementById('kpiBertF1'),
        svgChartContainer: document.getElementById('svgChartContainer'),
        benchmarkTableBody: document.getElementById('benchmarkTableBody'),
        
        // Gallery
        casesGridContainer: document.getElementById('casesGridContainer'),
        
        // Modal
        saveCaseModal: document.getElementById('saveCaseModal'),
        modalCaseTitle: document.getElementById('modalCaseTitle'),
        modalCaseTag: document.getElementById('modalCaseTag'),
        modalCaseText: document.getElementById('modalCaseText'),
        modalCaseNote: document.getElementById('modalCaseNote'),
        btnCloseModal: document.getElementById('btnCloseModal'),
        btnCancelSave: document.getElementById('btnCancelSave'),
        btnConfirmSave: document.getElementById('btnConfirmSave'),
        
        toastContainer: document.getElementById('toastContainer')
      }};

      /* ========================================================================
         工具函数：转义、提示、持久化
         ======================================================================== */
      function escapeHtml(str) {{
        if (!str) return '';
        return String(str)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#039;');
      }}

      function showToast(msg, icon = '✓') {{
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `<span>${{icon}}</span><span>${{escapeHtml(msg)}}</span>`;
        el.toastContainer.appendChild(toast);
        setTimeout(() => {{
          toast.style.opacity = '0';
          toast.style.transition = 'opacity 0.25s ease';
          setTimeout(() => toast.remove(), 250);
        }}, 2400);
      }}

      /* ========================================================================
         主题切换 (Theme Manager)
         ======================================================================== */
      function initTheme() {{
        const savedTheme = localStorage.getItem('macbert_ner_theme') || localStorage.getItem('macbert_workbench_theme') || 'dark';
        setTheme(savedTheme);
        el.themeToggleBtn.addEventListener('click', () => {{
          const current = document.documentElement.getAttribute('data-theme') || 'dark';
          setTheme(current === 'dark' ? 'light' : 'dark');
        }});
      }}

      function setTheme(theme) {{
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('macbert_ner_theme', theme);
        localStorage.setItem('macbert_workbench_theme', theme);
        if (theme === 'dark') {{
          el.themeIconSun.style.display = 'none';
          el.themeIconMoon.style.display = 'block';
        }} else {{
          el.themeIconSun.style.display = 'block';
          el.themeIconMoon.style.display = 'none';
        }}
        renderBenchmarkChart();
      }}

      /* ========================================================================
         后端连通性探测与自适应降级 (Graceful Fallback Engine)
         ======================================================================== */
      async function detectBackend() {{
        try {{
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 1200);
          const res = await fetch('/api/health', {{ signal: controller.signal }});
          clearTimeout(timer);
          if (res.ok) {{
            const data = await res.json();
            if (data.status === 'ok') {{
              state.isOnline = true;
              el.statusDot.className = 'status-dot online';
              el.statusText.textContent = '后端服务在线 (PyTorch CPU 实时推理)';
              el.offlineNoticeBanner.style.display = 'none';
              return;
            }}
          }}
        }} catch (err) {{
          // 连通失败，降级为离线模式
        }}

        state.isOnline = false;
        el.statusDot.className = 'status-dot';
        el.statusText.textContent = '离线演示模式 (本地静态推演引擎)';
        el.offlineNoticeBanner.style.display = 'flex';
      }}

      /* ========================================================================
         自定义用例管理系统 (Custom Cases System)
         ======================================================================== */
      const STORAGE_KEY_CASES = 'macbert_ner_custom_cases';

      function loadCustomCases() {{
        const stored = localStorage.getItem(STORAGE_KEY_CASES) || localStorage.getItem('macbert_ner_custom_cases_v2');
        if (stored) {{
          try {{
            state.customCases = JSON.parse(stored);
          }} catch (e) {{
            state.customCases = [];
          }}
        }} else {{
          // 首次加载初始化 3 个种子用例
          state.customCases = [
            {{
              id: 'case_seed_1',
              title: '国家新能源政策与产业协会',
              tag: '政府/组织机构',
              text: '工业和信息化部与中国汽车工业协会于北京亦庄共同商讨固态电池产业链技术标准。',
              note: '测试长机构复合名与部委行政实体',
              time: '预置用例'
            }},
            {{
              id: 'case_seed_2',
              title: '商业综合体与全球连锁品牌',
              tag: '公司/地址',
              text: '位于深圳市南山区科技园的华润万象天地吸引了特斯拉、耐克与星巴克首家概念店入驻。',
              note: '考察中英混合地标与多实体识别',
              time: '预置用例'
            }},
            {{
              id: 'case_seed_3',
              title: '文娱同名与角色消歧',
              tag: '电影/书籍/人名',
              text: '导演郭帆在上海影城执导了电影《流浪地球2》，原著由科幻作家刘慈欣创作。',
              note: '消歧实体跨度与细粒度属性',
              time: '预置用例'
            }}
          ];
          saveCustomCasesToStorage();
        }}
        renderCustomCases();
      }}

      function saveCustomCasesToStorage() {{
        localStorage.setItem(STORAGE_KEY_CASES, JSON.stringify(state.customCases));
        renderCustomCases();
      }}

      function renderCustomCases() {{
        el.customCasesBadge.textContent = state.customCases.length;
        if (state.customCases.length === 0) {{
          el.customCasesGrid.innerHTML = '<div style="font-size:0.8rem; color:var(--text-muted); grid-column:1/-1; text-align:center; padding:1.5rem 0;">暂无自定义用例，输入文本后点击“保存当前用例”即可添加</div>';
          return;
        }}

        el.customCasesGrid.innerHTML = state.customCases.map((c, idx) => `
          <div class="custom-case-item">
            <div class="custom-case-meta">
              <span class="custom-case-tag">${{escapeHtml(c.tag || '自定义')}}</span>
              <span class="custom-case-time">${{escapeHtml(c.time || '')}}</span>
            </div>
            <div style="font-weight:700; font-size:0.85rem; color:var(--text-main);">${{escapeHtml(c.title)}}</div>
            <div class="custom-case-text">${{escapeHtml(c.text)}}</div>
            ${{c.note ? `<div class="custom-case-note">📝 ${{escapeHtml(c.note)}}</div>` : ''}}
            <div class="custom-case-actions">
              <button class="btn btn-secondary btn-sm" onclick="window.__macbertApp.loadCaseText('${{c.id}}')">🚀 载入打靶</button>
              <button class="btn btn-secondary btn-sm" onclick="window.__macbertApp.editCase('${{c.id}}')">✏️ 编辑</button>
              <button class="btn btn-secondary btn-sm" style="color:var(--accent-danger);" onclick="window.__macbertApp.deleteCase('${{c.id}}')">🗑️</button>
            </div>
          </div>
        `).join('');
      }}

      function exportCasesJson() {{
        const blob = new Blob([JSON.stringify(state.customCases, null, 2)], {{ type: 'application/json' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `custom_ner_cases_${{Date.now()}}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('已成功导出用例集 JSON 文件');
      }}

      function importCasesJson(event) {{
        const file = event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function(e) {{
          try {{
            const list = JSON.parse(e.target.result);
            if (!Array.isArray(list)) throw new Error('JSON 格式必须为用例数组');
            let added = 0;
            list.forEach(item => {{
              if (item.text && item.title) {{
                state.customCases.unshift({{
                  id: 'case_imp_' + Math.random().toString(36).substr(2, 9),
                  title: item.title,
                  tag: item.tag || '导入用例',
                  text: item.text,
                  note: item.note || '',
                  time: '刚刚导入'
                }});
                added++;
              }}
            }});
            saveCustomCasesToStorage();
            showToast(`成功导入 ${{added}} 条测试用例！`);
          }} catch (err) {{
            alert('用例文件解析失败: ' + err.message);
          }}
        }};
        reader.readAsText(file);
        event.target.value = '';
      }}

      /* ========================================================================
         模块 1：⚡ 模型对决与在线打靶渲染 (Arena Runner)
         ======================================================================== */
      function initArena() {{
        // 字数统计与限制
        el.arenaInput.addEventListener('input', () => {{
          el.charCount.textContent = `${{el.arenaInput.value.length}} / 128 字`;
        }});

        // 清空
        el.btnClearInput.addEventListener('click', () => {{
          el.arenaInput.value = '';
          el.charCount.textContent = '0 / 128 字';
          el.arenaInput.focus();
        }});

        // 折叠自定义面板
        el.btnToggleCustomCases.addEventListener('click', () => {{
          const isHidden = el.customCasesPanel.style.display === 'none';
          el.customCasesPanel.style.display = isHidden ? 'block' : 'none';
        }});

        // 导出/导入
        el.btnExportCases.addEventListener('click', exportCasesJson);
        el.btnImportCases.addEventListener('click', () => el.importFileInput.click());
        el.importFileInput.addEventListener('change', importCasesJson);

        // 保存弹窗事件
        el.btnOpenSaveModal.addEventListener('click', () => {{
          const txt = el.arenaInput.value.trim();
          if (!txt) {{
            alert('请先在输入框中输入测试文本后再保存！');
            return;
          }}
          state.editingCaseId = null;
          const header = document.getElementById('modalCaseTitleHeader');
          if (header) header.textContent = '⭐ 保存为自定义测试用例';
          el.modalCaseText.value = txt;
          el.modalCaseTitle.value = txt.slice(0, 16) + (txt.length > 16 ? '...' : '');
          el.modalCaseTag.value = '在线测试';
          el.modalCaseNote.value = '';
          el.saveCaseModal.classList.add('active');
        }});

        el.btnCloseModal.addEventListener('click', () => {{
          state.editingCaseId = null;
          el.saveCaseModal.classList.remove('active');
        }});
        el.btnCancelSave.addEventListener('click', () => {{
          state.editingCaseId = null;
          el.saveCaseModal.classList.remove('active');
        }});
        el.btnConfirmSave.addEventListener('click', () => {{
          const title = el.modalCaseTitle.value.trim();
          const text = el.modalCaseText.value.trim();
          if (!title || !text) {{
            alert('标题与待测文本不能为空！');
            return;
          }}
          if (state.editingCaseId) {{
            const target = state.customCases.find(item => item.id === state.editingCaseId);
            if (target) {{
              target.title = title;
              target.tag = el.modalCaseTag.value.trim() || '自定义';
              target.text = text;
              target.note = el.modalCaseNote.value.trim();
              target.time = '刚刚更新';
            }}
            state.editingCaseId = null;
            saveCustomCasesToStorage();
            el.saveCaseModal.classList.remove('active');
            showToast('用例已成功更新');
          }} else {{
            state.customCases.unshift({{
              id: 'case_usr_' + Date.now(),
              title: title,
              tag: el.modalCaseTag.value.trim() || '自定义',
              text: text,
              note: el.modalCaseNote.value.trim(),
              time: '刚刚保存'
            }});
            saveCustomCasesToStorage();
            el.saveCaseModal.classList.remove('active');
            showToast('用例已成功持久化至本地用例库');
          }}
        }});

        // 底层 Token 探针折叠开关
        el.btnToggleProbe.addEventListener('click', () => {{
          const isShown = el.probeTableWrapper.style.display === 'block';
          el.probeTableWrapper.style.display = isShown ? 'none' : 'block';
          el.probeToggleArrow.textContent = isShown ? '▼ 展开探针' : '▲ 收起探针';
        }});

        // 渲染预设样本胶囊
        const presets = bundle.presets || [];
        el.presetPillsContainer.innerHTML = presets.map((p, i) => `
          <button class="preset-pill ${{i === 0 ? 'active' : ''}}" data-preset-id="${{p.id}}" title="${{p.desc}}">
            ${{p.title}}
          </button>
        `).join('');

        el.presetPillsContainer.addEventListener('click', (e) => {{
          const btn = e.target.closest('.preset-pill');
          if (!btn) return;
          document.querySelectorAll('.preset-pill').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          const pId = btn.getAttribute('data-preset-id');
          const found = presets.find(item => item.id === pId);
          if (found) {{
            el.arenaInput.value = found.text;
            el.charCount.textContent = `${{found.text.length}} / 128 字`;
            runArenaAnalysis(found.text, pId);
          }}
        }});

        // 打靶按钮
        el.btnRunArena.addEventListener('click', () => {{
          const text = el.arenaInput.value.trim();
          if (!text) {{
            alert('请输入待测试的中文文本！');
            return;
          }}
          runArenaAnalysis(text);
        }});

        // 默认载入第一个预设
        if (presets.length > 0) {{
          el.arenaInput.value = presets[0].text;
          el.charCount.textContent = `${{presets[0].text.length}} / 128 字`;
          runArenaAnalysis(presets[0].text, presets[0].id);
        }}
      }}

      // 执行对决打靶分析（核心推理路由）
      async function runArenaAnalysis(text, presetId = null) {{
        setLoading(true);

        try {{
          let data = null;

          // 1. 若在线，向后端发起请求
          if (state.isOnline) {{
            try {{
              const res = await fetch('/api/predict', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ text: text, model: 'both' }})
              }});
              if (res.ok) {{
                const resJson = await res.json();
                if (resJson.success) {{
                  data = resJson;
                }}
              }}
            }} catch (netErr) {{
              console.warn('后端推理请求失败，自动切换离线推演引擎:', netErr);
            }}
          }}

          // 2. 若离线或后端失败，走本地推演引擎
          if (!data) {{
            data = runOfflineSimulation(text, presetId);
          }}

          state.lastPredictResult = data;
          renderArenaResults(text, data);
        }} catch (err) {{
          console.error('Arena 执行异常:', err);
          alert('打靶分析失败: ' + err.message);
        }} finally {{
          setLoading(false);
        }}
      }}

      function setLoading(isLoading) {{
        el.btnRunArena.disabled = isLoading;
        if (isLoading) {{
          el.btnRunIcon.textContent = '⏳';
          el.btnRunText.textContent = '正在同屏前向推理...';
        }} else {{
          el.btnRunIcon.textContent = '⚡';
          el.btnRunText.textContent = '开始模型对决分析';
        }}
      }}

      // 本地离线静态推演引擎
      function runOfflineSimulation(text, presetId) {{
        const predictions = bundle.preset_predictions || {{}};

        // 检查是否有预置推演记录
        if (presetId && predictions[presetId]) {{
          return predictions[presetId];
        }}
        if (predictions[text]) {{
          return predictions[text];
        }}

        // 启发式匹配已有错例
        for (const k in predictions) {{
          if (predictions[k] && (k === text || text.includes(k) || k.includes(text))) {{
            return predictions[k];
          }}
        }}

        // 通用兜底模拟推演
        return simulateArbitraryText(text);
      }}

      function simulateArbitraryText(text) {{
        const meta = bundle.category_meta || {{}};
        // 简单词典匹配提取模拟
        const mockDict = [
          {{ w: '中国汽车工业协会', cat: 'organization' }},
          {{ w: '莫斯科中央陆军', cat: 'organization' }},
          {{ w: '星巴克', cat: 'company' }},
          {{ w: '苹果公司', cat: 'company' }},
          {{ w: '腾讯', cat: 'company' }},
          {{ w: '王者荣耀', cat: 'game' }},
          {{ w: '地下城与勇士', cat: 'game' }},
          {{ w: '三体', cat: 'book' }},
          {{ w: '张伟', cat: 'name' }},
          {{ w: '李雷', cat: 'name' }},
          {{ w: '王华', cat: 'name' }},
          {{ w: '上海市', cat: 'address' }},
          {{ w: '北京市', cat: 'address' }},
          {{ w: '颐和园', cat: 'scene' }},
          {{ w: '教授', cat: 'position' }},
          {{ w: '行长', cat: 'position' }}
        ];

        const mEntities = [];
        mockDict.forEach(item => {{
          let idx = text.indexOf(item.w);
          while (idx !== -1) {{
            mEntities.push({{
              start: idx,
              end: idx + item.w.length - 1,
              category: item.cat,
              category_cn: meta[item.cat]?.cn || item.cat,
              color: 'var(--text-main)',
              text: item.w,
              confidence: 0.965
            }});
            idx = text.indexOf(item.w, idx + item.w.length);
          }}
        }});

        const bEntities = JSON.parse(JSON.stringify(mEntities));

        const charTokens = Array.from(text).map((ch, i) => ({{
          index: i,
          char: ch,
          bio: 'O',
          confidence: 0.98
        }}));

        return {{
          results: {{
            macbert: {{
              model_key: 'macbert',
              model_name: 'chinese-macbert-base',
              latency_ms: 12.5,
              entity_count: mEntities.length,
              entities: mEntities,
              tokens: charTokens
            }},
            bert: {{
              model_key: 'bert',
              model_name: 'bert-base-chinese',
              latency_ms: 11.8,
              entity_count: bEntities.length,
              entities: bEntities,
              tokens: charTokens
            }}
          }},
          diff: {{
            has_diff: false,
            status_tag: 'identical',
            summary: '✨ 离线推演引擎完成模拟标注，两模型在预设模式下保持平稳。',
            common_entities: mEntities,
            macbert_unique: [],
            bert_unique: [],
            boundary_repairs: [],
            category_conflicts: [],
            diff_token_count: 0
          }}
        }};
      }}

      // 渲染打靶对比结果
      function renderArenaResults(text, data) {{
        const mRes = data.results?.macbert || {{ entities: [], tokens: [], latency_ms: 0 }};
        const bRes = data.results?.bert || {{ entities: [], tokens: [], latency_ms: 0 }};
        const diff = data.diff || {{}};

        // 1. 延迟与数量
        el.macbertLatency.textContent = `推理: ${{mRes.latency_ms}} ms`;
        el.macbertCount.textContent = `实体: ${{mRes.entities.length}} 个`;
        el.bertLatency.textContent = `推理: ${{bRes.latency_ms}} ms`;
        el.bertCount.textContent = `实体: ${{bRes.entities.length}} 个`;

        // 2. 渲染高亮文本框
        el.macbertVisualBox.innerHTML = renderHighlightedSpanText(text, mRes.entities);
        el.bertVisualBox.innerHTML = renderHighlightedSpanText(text, bRes.entities);

        // 3. 渲染结构化实体 Chips 列表
        el.macbertEntitiesList.innerHTML = renderEntitiesChips(mRes.entities);
        el.bertEntitiesList.innerHTML = renderEntitiesChips(bRes.entities);

        // 4. 渲染差异报告视窗
        renderDiffAlert(diff);

        // 5. 渲染底层 Token 探针
        renderTokenProbe(text, mRes.tokens, bRes.tokens);
      }}

      // 将文本按实体切分并插入高亮标签
      function renderHighlightedSpanText(text, entities) {{
        if (!entities || entities.length === 0) {{
          return escapeHtml(text) || '<span style="color:var(--text-muted)">（未识别到命名实体）</span>';
        }}
        const sorted = [...entities].sort((a, b) => a.start - b.start);
        let html = '';
        let lastIdx = 0;

        for (const ent of sorted) {{
          if (ent.start < lastIdx) continue;
          if (ent.start > lastIdx) {{
            html += escapeHtml(text.slice(lastIdx, ent.start));
          }}
          const cat = ent.category;
          const catCn = ent.category_cn || cat;
          const confStr = (ent.confidence * 100).toFixed(1) + '%';
          const spanText = escapeHtml(text.slice(ent.start, ent.end + 1));

          html += `<span class="entity-span badge-${{cat}}" title="${{catCn}} [${{ent.start}}, ${{ent.end}}] ${{confStr}}">
            <span class="ent-text">${{spanText}}</span>
            <span class="ent-label">${{catCn}}</span>
            <span class="ent-tooltip">
              <strong>【${{catCn}} ${{cat}}】</strong><br>
              文本: "${{spanText}}"<br>
              区间: [${{ent.start}}, ${{ent.end}}] (${{spanText.length}}字)<br>
              置信度: ${{confStr}}
            </span>
          </span>`;
          lastIdx = ent.end + 1;
        }}

        if (lastIdx < text.length) {{
          html += escapeHtml(text.slice(lastIdx));
        }}
        return html;
      }}

      // 渲染右侧实体明细条目
      function renderEntitiesChips(entities) {{
        if (!entities || entities.length === 0) {{
          return '<div style="font-size:0.8rem; color:var(--text-muted); padding:0.5rem 0;">未检出任何实体</div>';
        }}
        return entities.map(e => `
          <div class="entity-chip-row">
            <div class="entity-chip-left">
              <span class="entity-span badge-${{e.category}}" style="margin:0; font-size:0.75rem; padding:0.1rem 0.35rem;">${{escapeHtml(e.category_cn || e.category)}}</span>
              <strong style="color:var(--text-main); font-size:0.82rem;">${{escapeHtml(e.text)}}</strong>
              <span class="entity-chip-span">[${{e.start}}:${{e.end}}]</span>
            </div>
            <div class="entity-chip-right">
              <div class="conf-bar-bg" title="置信度: ${{(e.confidence * 100).toFixed(1)}}%">
                <div class="conf-bar-fill" style="width:${{(e.confidence * 100).toFixed(0)}}%;"></div>
              </div>
              <span class="conf-percent">${{(e.confidence * 100).toFixed(1)}}%</span>
            </div>
          </div>
        `).join('');
      }}

      // 渲染差异分析视窗
      function renderDiffAlert(diff) {{
        if (!diff || !diff.has_diff) {{
          el.diffAlertContainer.innerHTML = `
            <div class="diff-alert-card identical">
              <div class="diff-header">
                <div class="diff-title" style="color:var(--text-main);">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
                  两模型预测完全一致 (No Discrepancy)
                </div>
                <span class="stat-chip" style="color:var(--text-secondary); border-color:var(--border-subtle);">一致率: 100%</span>
              </div>
              <div class="diff-desc">${{escapeHtml(diff.summary || 'BERT 与 MacBERT 对当前句子的所有实体边界切分与类别判定完全吻合。')}}</div>
            </div>
          `;
          return;
        }}

        let repairsHtml = '';
        if (diff.boundary_repairs && diff.boundary_repairs.length > 0) {{
          repairsHtml = diff.boundary_repairs.map(r => `
            <div class="diff-detail-item">
              <span style="color:var(--text-main); font-weight:700;">[MacBERT 全词修复]</span>
              <div>
                <strong>${{escapeHtml(r.macbert_entity.text)}}</strong> (跨度 [${{r.macbert_entity.start}}:${{r.macbert_entity.end}}], 完整识别为 ${{r.macbert_entity.category_cn}})
                <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.15rem;">
                  BERT 发生碎片截断，切散为: ${{r.bert_fragments.map(f => `「${{escapeHtml(f.text)}}」(${{f.category_cn}})`).join(' + ')}}
                </div>
              </div>
            </div>
          `).join('');
        }}

        let conflictsHtml = '';
        if (diff.category_conflicts && diff.category_conflicts.length > 0) {{
          conflictsHtml = diff.category_conflicts.map(c => `
            <div class="diff-detail-item">
              <span style="color:var(--text-secondary); font-weight:700;">[细粒度消歧差异]</span>
              <div>${{escapeHtml(c.description)}}</div>
            </div>
          `).join('');
        }}

        let uniqueHtml = '';
        const coveredMStarts = new Set((diff.boundary_repairs || []).map(r => r.macbert_entity?.start));
        const coveredBStarts = new Set((diff.boundary_repairs || []).flatMap(r => (r.bert_fragments || []).map(f => f.start)));

        if (diff.macbert_unique && diff.macbert_unique.length > 0) {{
          diff.macbert_unique.forEach(u => {{
            if (!coveredMStarts.has(u.start)) {{
              uniqueHtml += `
                <div class="diff-detail-item">
                  <span style="color:var(--text-main); font-weight:700;">[MacBERT 独有召回]</span>
                  <div>
                    <strong>${{escapeHtml(u.text)}}</strong> 判定为【${{escapeHtml(u.category_cn || u.category)}}】(置信度 ${{(u.confidence * 100).toFixed(1)}}%)，跨度 [${{u.start}}:${{u.end}}]。BERT 未能检出此实体。
                  </div>
                </div>
              `;
            }}
          }});
        }}

        if (diff.bert_unique && diff.bert_unique.length > 0) {{
          diff.bert_unique.forEach(u => {{
            if (!coveredBStarts.has(u.start)) {{
              uniqueHtml += `
                <div class="diff-detail-item">
                  <span style="color:var(--text-muted); font-weight:700;">[BERT 独有标注]</span>
                  <div>
                    <strong>${{escapeHtml(u.text)}}</strong> 判定为【${{escapeHtml(u.category_cn || u.category)}}】(置信度 ${{(u.confidence * 100).toFixed(1)}}%)，跨度 [${{u.start}}:${{u.end}}]。MacBERT 判定为非实体。
                  </div>
                </div>
              `;
            }}
          }});
        }}

        el.diffAlertContainer.innerHTML = `
          <div class="diff-alert-card">
            <div class="diff-header">
              <div class="diff-title" style="color:var(--text-main);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                模型对比分歧报告 (Diff Mode Active)
              </div>
              <span class="stat-chip" style="color:var(--text-main); border-color:var(--border-subtle);">
                分歧 Token: ${{diff.diff_token_count || 0}} 字 (${{diff.diff_token_ratio || 0}}%)
              </span>
            </div>
            <div class="diff-desc">${{escapeHtml(diff.summary)}}</div>
            <div class="diff-details-list">
              ${{repairsHtml}}
              ${{conflictsHtml}}
              ${{uniqueHtml}}
            </div>
          </div>
        `;
      }}

      // 渲染底层 Token 探针
      function renderTokenProbe(text, mTokens, bTokens) {{
        const maxLen = Math.max(mTokens.length, bTokens.length, text.length);
        let rows = '';
        let diffCount = 0;

        for (let i = 0; i < maxLen; i++) {{
          const ch = text[i] || '';
          const mTok = mTokens[i] || {{ bio: 'O', confidence: 0 }};
          const bTok = bTokens[i] || {{ bio: 'O', confidence: 0 }};
          const isDiff = mTok.bio !== bTok.bio;
          if (isDiff) diffCount++;

          const mBioClass = mTok.bio.startsWith('B-') ? 'bio-B' : (mTok.bio.startsWith('I-') ? 'bio-I' : 'bio-O');
          const bBioClass = bTok.bio.startsWith('B-') ? 'bio-B' : (bTok.bio.startsWith('I-') ? 'bio-I' : 'bio-O');

          rows += `
            <tr class="${{isDiff ? 'row-diff' : ''}}">
              <td style="color:var(--text-muted); font-family:var(--font-mono);">${{i}}</td>
              <td><strong style="color:var(--text-main); font-size:0.95rem;">${{escapeHtml(ch)}}</strong></td>
              <td><span class="bio-tag ${{bBioClass}}">${{escapeHtml(bTok.bio)}}</span></td>
              <td style="font-family:var(--font-mono); color:var(--text-muted);">${{(bTok.confidence * 100).toFixed(1)}}%</td>
              <td><span class="bio-tag ${{mBioClass}}">${{escapeHtml(mTok.bio)}}</span></td>
              <td style="font-family:var(--font-mono); color:var(--text-muted);">${{(mTok.confidence * 100).toFixed(1)}}%</td>
              <td>
                ${{isDiff 
                  ? '<span style="color:var(--text-main); font-weight:700;">● 分歧</span>' 
                  : '<span style="color:var(--text-muted);">○ 一致</span>'}}
              </td>
            </tr>
          `;
        }}

        el.probeTableBody.innerHTML = rows;
        if (diffCount > 0) {{
          el.probeDiffCountBadge.style.display = 'inline-block';
          el.probeDiffCountBadge.textContent = `${{diffCount}} 处分歧`;
        }} else {{
          el.probeDiffCountBadge.style.display = 'none';
        }}
      }}

      /* ========================================================================
         模块 2：💻 复现代码深度透视 (Code Walkthrough)
         ======================================================================== */
      function initCodeWalkthrough() {{
        const codeData = bundle.code || {{ files: [] }};
        const files = codeData.files || [];

        el.codeFilesNav.innerHTML = files.map((f, i) => `
          <button class="code-tab-btn ${{f.id === state.activeCodeFile ? 'active' : ''}}" data-code-id="${{f.id}}">
            📄 ${{f.filename}}
          </button>
        `).join('');

        el.codeFilesNav.addEventListener('click', (e) => {{
          const btn = e.target.closest('.code-tab-btn');
          if (!btn) return;
          const fid = btn.getAttribute('data-code-id');
          state.activeCodeFile = fid;
          document.querySelectorAll('.code-tab-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          renderCodeFile(fid);
        }});

        el.btnCopyCode.addEventListener('click', () => {{
          const txt = el.codeSourceDisplay.textContent;
          navigator.clipboard.writeText(txt).then(() => {{
            showToast('代码已复制到剪贴板！');
          }}).catch(() => {{
            showToast('复制失败，请手动选择复制');
          }});
        }});

        if (files.length > 0) {{
          renderCodeFile(files[0].id);
        }}
      }}

      function renderCodeFile(fileId) {{
        const codeData = bundle.code || {{ files: [] }};
        const fileObj = (codeData.files || []).find(f => f.id === fileId);
        if (!fileObj) return;

        el.currentCodeFilename.textContent = fileObj.filename;
        el.codeSourceDisplay.textContent = fileObj.source_code || '# No source';

        // 渲染考点批注
        const anns = fileObj.annotations || [];
        el.codeAnnotationsContainer.innerHTML = anns.map(a => `
          <div class="annotation-card">
            <div class="annotation-title">🎯 ${{escapeHtml(a.title)}}</div>
            <pre class="annotation-snippet"><code>${{escapeHtml(a.code_snippet)}}</code></pre>
            <div class="annotation-text">${{escapeHtml(a.explanation)}}</div>
          </div>
        `).join('');
      }}

      /* ========================================================================
         模块 3：📊 实测学术指标大盘 (Benchmark Dashboard)
         ======================================================================== */
      function initBenchmark() {{
        document.querySelectorAll('.chart-btn').forEach(btn => {{
          btn.addEventListener('click', (e) => {{
            document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeMetric = btn.getAttribute('data-metric');
            renderBenchmarkChart();
          }});
        }});

        renderBenchmarkChart();
        renderBenchmarkTable();
      }}

      function renderBenchmarkChart() {{
        const metricsData = bundle.metrics || {{}};
        const comp = metricsData.categories_comparison || [];
        if (comp.length === 0) return;

        const metricKey = state.activeMetric; // f1, precision, recall
        const isDark = document.documentElement.getAttribute('data-theme') !== 'light';

        const chartWidth = 960;
        const chartHeight = 360;
        const paddingLeft = 100;
        const paddingBottom = 40;
        const paddingTop = 30;
        const paddingRight = 40;

        const plotWidth = chartWidth - paddingLeft - paddingRight;
        const plotHeight = chartHeight - paddingTop - paddingBottom;
        const barGroupWidth = plotWidth / comp.length;
        const barWidth = Math.min(barGroupWidth * 0.35, 24);

        let svg = `<svg viewBox="0 0 ${{chartWidth}} ${{chartHeight}}" style="width:100%; height:auto; overflow:visible;">`;

        // 绘制 Y 轴刻度线 (0%, 25%, 50%, 75%, 100%) - 极简灰度网格
        const gridColor = isDark ? '#262626' : '#e5e5e5';
        const labelColor = isDark ? '#a3a3a3' : '#525252';
        const macbertBarColor = isDark ? '#ffffff' : '#000000';
        const bertBarColor = isDark ? '#525252' : '#a3a3a3';

        for (let pct = 0; pct <= 100; pct += 25) {{
          const y = paddingTop + plotHeight * (1 - pct / 100);
          svg += `<line x1="${{paddingLeft}}" y1="${{y}}" x2="${{chartWidth - paddingRight}}" y2="${{y}}" stroke="${{gridColor}}" stroke-dasharray="2,2" />`;
          svg += `<text x="${{paddingLeft - 12}}" y="${{y + 4}}" fill="${{labelColor}}" font-size="11" text-anchor="end" font-family="var(--font-mono)">${{pct}}%</text>`;
        }}

        // 绘制每个类别的双柱形 (MacBERT vs BERT)
        comp.forEach((item, idx) => {{
          const groupCenterX = paddingLeft + (idx + 0.5) * barGroupWidth;
          const bVal = item.bert[metricKey] || 0;
          const mVal = item.macbert[metricKey] || 0;

          const bHeight = (bVal / 100) * plotHeight;
          const mHeight = (mVal / 100) * plotHeight;

          const bX = groupCenterX - barWidth - 2;
          const mX = groupCenterX + 2;

          const bY = paddingTop + (plotHeight - bHeight);
          const mY = paddingTop + (plotHeight - mHeight);

          // BERT 柱体 (中灰基准)
          svg += `<rect x="${{bX}}" y="${{bY}}" width="${{barWidth}}" height="${{bHeight}}" rx="2" fill="${{bertBarColor}}">
            <title>BERT (${{item.name_cn}}): ${{bVal.toFixed(1)}}%</title>
          </rect>`;

          // MacBERT 柱体 (高对比纯白/纯黑)
          svg += `<rect x="${{mX}}" y="${{mY}}" width="${{barWidth}}" height="${{mHeight}}" rx="2" fill="${{macbertBarColor}}">
            <title>MacBERT (${{item.name_cn}}): ${{mVal.toFixed(1)}}%</title>
          </rect>`;

          // X 轴类别标签
          svg += `<text x="${{groupCenterX}}" y="${{chartHeight - paddingBottom + 18}}" fill="${{labelColor}}" font-size="11" font-weight="600" text-anchor="middle">${{item.name_cn}}</text>`;
          svg += `<text x="${{groupCenterX}}" y="${{chartHeight - paddingBottom + 32}}" fill="${{labelColor}}" font-size="9" text-anchor="middle" font-family="var(--font-mono)">${{item.category}}</text>`;

          // 增益差异小标签 (极简黑白灰高对比度)
          const delta = (mVal - bVal).toFixed(1);
          const deltaColor = isDark ? (delta > 0 ? '#ffffff' : '#737373') : (delta > 0 ? '#000000' : '#8c8c8c');
          const deltaSign = delta > 0 ? `+${{delta}}` : `${{delta}}`;
          svg += `<text x="${{groupCenterX}}" y="${{Math.min(bY, mY) - 6}}" fill="${{deltaColor}}" font-size="10" font-weight="700" text-anchor="middle" font-family="var(--font-mono)">${{deltaSign}}%</text>`;
        }});

        // 图例 (Legend)
        svg += `
          <g transform="translate(${{chartWidth - paddingRight - 220}}, 10)">
            <rect x="0" y="0" width="12" height="12" rx="2" fill="${{macbertBarColor}}" />
            <text x="18" y="10" fill="${{labelColor}}" font-size="11" font-weight="600">Chinese-MacBERT</text>
            <rect x="130" y="0" width="12" height="12" rx="2" fill="${{bertBarColor}}" />
            <text x="148" y="10" fill="${{labelColor}}" font-size="11" font-weight="600">BERT-base</text>
          </g>
        `;

        svg += `</svg>`;
        el.svgChartContainer.innerHTML = svg;
      }}

      function renderBenchmarkTable() {{
        const metricsData = bundle.metrics || {{}};
        const comp = metricsData.categories_comparison || [];
        el.benchmarkTableBody.innerHTML = comp.map(item => `
          <tr>
            <td>
              <span class="entity-span badge-${{item.category}}" style="font-size:0.75rem; margin:0; padding:0.1rem 0.35rem;">
                ${{item.name_cn}} (${{item.category}})
              </span>
            </td>
            <td style="font-family:var(--font-mono);">${{item.support}} 个跨度</td>
            <td style="font-family:var(--font-mono);">
              P: ${{item.bert.precision}}% / R: ${{item.bert.recall}}% / <span style="color:var(--text-secondary); font-weight:600;">F1: ${{item.bert.f1}}%</span>
            </td>
            <td style="font-family:var(--font-mono);">
              P: ${{item.macbert.precision}}% / R: ${{item.macbert.recall}}% / <strong style="color:var(--text-main);">F1: ${{item.macbert.f1}}%</strong>
            </td>
            <td>
              <span class="kpi-delta" style="font-family:var(--font-mono); font-size:0.8rem; border:1px solid var(--border-subtle); color:var(--text-main); font-weight:700;">
                ${{item.delta.f1 > 0 ? '+' : ''}}${{item.delta.f1}}%
              </span>
            </td>
            <td>
              ${{item.delta.f1 >= 1.0 
                ? '<span style="color:var(--text-main); font-weight:700;">★ 显著优势增益</span>' 
                : (item.delta.f1 > 0 ? '<span style="color:var(--text-secondary);">稳步提升</span>' : '<span style="color:var(--text-muted);">性能基本持平</span>')}}
            </td>
          </tr>
        `).join('');
      }}

      /* ========================================================================
         模块 4：🔍 真实错例深度展厅 (Bad Case Gallery)
         ======================================================================== */
      function initGallery() {{
        document.querySelectorAll('.gallery-filter-btn').forEach(btn => {{
          btn.addEventListener('click', () => {{
            document.querySelectorAll('.gallery-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeFilter = btn.getAttribute('data-filter');
            renderGalleryCases();
          }});
        }});

        renderGalleryCases();
      }}

      function renderGalleryCases() {{
        const casesData = bundle.cases || {{}};
        const catA = casesData.category_a_macbert_advantages || [];
        const catB = casesData.category_b_hard_cases || [];

        let list = [];
        if (state.activeFilter === 'advantages') {{
          list = catA;
        }} else if (state.activeFilter === 'hard') {{
          list = catB;
        }} else {{
          list = [...catA, ...catB];
        }}

        if (list.length === 0) {{
          el.casesGridContainer.innerHTML = '<div style="font-size:0.85rem; color:var(--text-muted); padding:2rem 0; text-align:center;">暂无案例</div>';
          return;
        }}

        el.casesGridContainer.innerHTML = list.map(c => `
          <div class="case-card">
            <div class="case-top-meta">
              <span class="case-id-badge">${{escapeHtml(c.case_id)}}</span>
              <span class="stat-chip">${{escapeHtml(c.primary_category || '综合')}}</span>
            </div>
            <div class="case-title">${{escapeHtml(c.pattern_name || '典型中文 NER 案例分析')}}</div>
            <div class="case-text-box">“${{escapeHtml(c.text)}}”</div>
            
            <div class="comparison-box">
              <div class="comp-row">
                <span class="comp-label" style="color:var(--text-main); font-weight:700;">🎯 真实标注:</span>
                <div class="comp-entities">
                  ${{renderEntitiesInlineBadges(c.ground_truth)}}
                </div>
              </div>
              <div class="comp-row">
                <span class="comp-label" style="color:var(--text-muted);">BERT预测:</span>
                <div class="comp-entities">
                  ${{renderEntitiesInlineBadges(c.bert_prediction)}}
                </div>
              </div>
              <div class="comp-row">
                <span class="comp-label" style="color:var(--text-main); font-weight:700;">MacBERT:</span>
                <div class="comp-entities">
                  ${{renderEntitiesInlineBadges(c.macbert_prediction)}}
                </div>
              </div>
            </div>

            <div class="case-reason">
              <strong>💡 机理剖析:</strong> ${{escapeHtml(c.technical_reason || c.diff_analysis || '')}}
            </div>

            <div class="case-card-footer">
              <button class="btn btn-primary btn-sm" onclick="window.__macbertApp.loadCaseToArena('${{escapeHtml(encodeURIComponent(c.text).replace(/'/g, '%27'))}}')">
                🚀 载入工作台现场重测
              </button>
            </div>
          </div>
        `).join('');
      }}

      function renderEntitiesInlineBadges(ents) {{
        if (!ents || ents.length === 0) {{
          return '<span style="color:var(--text-muted); font-size:0.75rem;">(无实体)</span>';
        }}
        return ents.map(e => `
          <span class="entity-span badge-${{e.category}}" style="font-size:0.72rem; padding:0.1rem 0.35rem; margin:0;">
            ${{escapeHtml(e.text)}} (${{e.category}})
          </span>
        `).join('');
      }}

      /* ========================================================================
         全局 Tab 切换联动
         ======================================================================== */
      function initTabNavigation() {{
        document.querySelectorAll('.tab-btn').forEach(btn => {{
          btn.addEventListener('click', () => {{
            const tabId = btn.getAttribute('data-tab');
            switchTab(tabId);
          }});
        }});
      }}

      function switchTab(tabId) {{
        state.activeTab = tabId;
        document.querySelectorAll('.tab-btn').forEach(b => {{
          b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
        }});
        document.querySelectorAll('.tab-pane').forEach(p => {{
          p.classList.toggle('active', p.id === tabId);
        }});
        if (tabId === 'tab-benchmark') {{
          renderBenchmarkChart();
        }}
      }}

      /* ========================================================================
         对外暴露挂载（供 HTML 内 onclick 调用）
         ======================================================================== */
      window.__macbertApp = {{
        loadCaseText: function(caseId) {{
          const c = state.customCases.find(item => item.id === caseId);
          if (c) {{
            el.arenaInput.value = c.text;
            el.charCount.textContent = `${{c.text.length}} / 128 字`;
            switchTab('tab-arena');
            runArenaAnalysis(c.text);
            showToast(`已载入自定义用例: ${{c.title}}`);
          }}
        }},
        editCase: function(caseId) {{
          const c = state.customCases.find(item => item.id === caseId);
          if (!c) return;
          state.editingCaseId = caseId;
          const header = document.getElementById('modalCaseTitleHeader');
          if (header) header.textContent = '✏️ 编辑自定义测试用例';
          el.modalCaseTitle.value = c.title || '';
          el.modalCaseTag.value = c.tag || '';
          el.modalCaseText.value = c.text || '';
          el.modalCaseNote.value = c.note || '';
          el.saveCaseModal.classList.add('active');
        }},
        deleteCase: function(caseId) {{
          if (confirm('确认删除该测试用例吗？')) {{
            state.customCases = state.customCases.filter(item => item.id !== caseId);
            saveCustomCasesToStorage();
            showToast('用例已删除');
          }}
        }},
        loadCaseToArena: function(encodedText) {{
          const text = decodeURIComponent(encodedText);
          el.arenaInput.value = text;
          el.charCount.textContent = `${{text.length}} / 128 字`;
          switchTab('tab-arena');
          runArenaAnalysis(text);
          window.scrollTo({{ top: 0, behavior: 'smooth' }});
          showToast('已载入错例至工作台对决区');
        }}
      }};

      /* ========================================================================
         应用初始化启动
         ======================================================================== */
      function main() {{
        initTheme();
        initTabNavigation();
        initArena();
        initCodeWalkthrough();
        initBenchmark();
        initGallery();
        loadCustomCases();
        detectBackend();
      }}

      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', main);
      }} else {{
        main();
      }}

    }})();
  </script>
</body>
</html>
'''

    # 1. 写入 templates/index.html (供 Flask 渲染托管)
    template_path = os.path.join(ROOT, "templates", "index.html")
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("[OK] Successfully generated Flask template: " + template_path)

    # 2. 写入根目录 workbench.html (供脱机直接双击离线浏览)
    workbench_path = os.path.join(ROOT, "workbench.html")
    with open(workbench_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("[OK] Successfully generated Standalone Offline workbench: " + workbench_path)

if __name__ == "__main__":
    build()
