# MetalTile Perf Planning Board

Interactive kanban board for tracking MetalTile performance research ideas. Drag-and-drop cards between columns, search/filter, and click cards for details.

## Live Board

👉 **[View the board](https://wafflehaus.github.io/metaltile-planning/)** (after GitHub Pages is enabled)

## Setup

### 1. Create the repo

Go to https://github.com/new and create `wafflehaus/metaltile-planning` as a **public** repo.

### 2. Push these files

```bash
cd metaltile-planning
git init
git remote add origin https://github.com/wafflehaus/metaltile-planning.git
git add .
git commit -m "Initial kanban board"
git push -u origin main
```

### 3. Enable GitHub Pages

1. Go to **Settings → Pages** in the repo.
2. Source: **Deploy from a branch**.
3. Branch: `main` / `root`.
4. Save. The board will be live at `https://wafflehaus.github.io/metaltile-planning/` within a minute.

## Features

- **Drag & drop** cards between columns (Blocked → Feasible → Done → etc.)
- **State persistence** — board layout saves to `localStorage` (per browser)
- **Search** — filter cards by name, notes, or ID
- **Category filter** — show only Quick-wins, Codegen passes, Moonshots, etc.
- **Detail modal** — click any card to see full metadata + link to assessment file
- **Responsive** — works on mobile (stacks columns vertically)

## Data Source

`data.json` contains all 65 assessed ideas from the `perf-research/` hopper. To regenerate:

```bash
# From the main metaltile repo:
python3 scripts/export_ideas.py > ../metaltile-planning/data.json
cd ../metaltile-planning
git add data.json && git commit -m "sync ideas" && git push
```

## Columns

| Column | Meaning |
|--------|---------|
| 🔴 Blocked | Prerequisite missing or idea ill-formed |
| ⚠️ Feasible | Actionable — ready to implement or experiment |
| ⚪ No-op / Marginal | Already implemented or not worth pursuing |
| 🟢 Done | Committed and benchmarked |
| ⚫ Abandoned | Discarded with documented reason |

## Contributing

When you move a card in the browser, its new position is saved locally. To share the updated board state with the team, export the state from browser DevTools:

```js
// In browser console on the board page:
localStorage.getItem('metaltile-planning-state')
```

Copy that JSON and commit it as `state.json` if you want to version-control board snapshots.
