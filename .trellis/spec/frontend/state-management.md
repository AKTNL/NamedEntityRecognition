# State Management

> How state is managed in the Interactive Workbench frontend.

---

## Overview

The Workbench is implemented as a modern zero-dependency vanilla JS single-page application that works seamlessly in both online mode (served by Flask) and standalone offline mode (`workbench.html`). State is centralized in a clean immutable/reactive pattern with local persistence.

---

## State Categories

1. **Runtime Server State (`state.isOnline`)**:
   - Detected at startup via probe to `/api/health`.
   - If connected, API requests fetch live results from the server.
   - If disconnected or running via `file://`, the app automatically falls back to the embedded `window.__NER_BUNDLE__` for instant static deduction and demonstrations.

2. **UI Navigation & View State**:
   - `state.activeTab`: Currently visible module (`tab-arena`, `tab-code`, `tab-benchmark`, `tab-gallery`).
   - `state.activeMetric`: Benchmark chart view mode (`f1`, `precision`, `recall`).
   - `state.activeFilter`: Bad case gallery category filter (`all`, `cat_a`, `cat_b`).
   - `state.activeCodeFile`: Active code file tab in walkthrough (`dataset`, `train`, `analyze_cases`).

3. **Persistent User State (LocalStorage)**:
   - `macbert_ner_theme`: Selected theme (`theme-dark` vs `theme-light`).
   - `macbert_ner_custom_cases`: Array of custom test cases created, edited, or imported by the user.

---

## LocalStorage Synchronization Contract

- Key: `macbert_ner_custom_cases`
- Schema:
  ```json
  [
    {
      "id": "case_usr_1710892800000",
      "title": "String (required)",
      "tag": "String (optional, default: '自定义')",
      "text": "String (required, <= 128 chars)",
      "note": "String (optional)",
      "time": "String timestamp or label"
    }
  ]
  ```
- Mutations: Any add, edit, delete, or import operation must update `state.customCases` in-memory and synchronously call `saveCustomCasesToStorage()`.

---

## Common Mistakes

- Forgetting to escape user-supplied text when rendering HTML attributes or dynamic spans (can cause XSS or broken markup when single quotes or special characters exist).
- Directly writing to `localStorage` without updating the UI badge counters.

