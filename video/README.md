# Project video: H&M Fashion Recommender (week 3 of 4)

An 86.5-second explainer built with [Remotion](https://www.remotion.dev/) and the series' shared light-theme
kit (`src/kit/`). The customer on screen is a real demo customer from the serving bundle. Their purchases,
the 12 recommendations and the reasons come from `reports/serving/`. Every number is read from the project's
reports (`src/data.ts` imports `reports/serving/metrics.json`, `reports/feature_importance.json`,
`reports/kaggle_score.json` and `reports/monitoring.json`). The few that exist only in `README.md`/`RESULTS.md`
are kept together in `README_NUM` in `src/data.ts`, each with its source.

| Output | Size | Use |
|---|---|---|
| `out/hm-wide.mp4` | 1920 × 1080 | YouTube, GitHub README, portfolio site |
| `out/hm-square.mp4` | 1080 × 1080 | LinkedIn feed |

## Rebuild

From the project root (it needs `reports/serving/`, `data/interim/` and `data/thumbs/`, so run training and
download the demo thumbnails first, as described in the main README):

```bash
.venv/Scripts/python video/scripts/extract.py   # -> video/public/data/demo.json + up to 16 thumbnails
cd video
npm install
npm run studio                                 # live preview
node scripts/stills.mjs Wide 180 870 1325      # review stills -> out/stills/
npm run render:wide
npm run render:square
```

## Demo recording

`public/demo.mp4` is a screen recording of the real Streamlit demo (`app/dashboard.py`), made in headless
Chrome with the light colour scheme:

```bash
# project root, in a second terminal
UV_CACHE_DIR=D:/uv-cache uv run --no-sync streamlit run app/dashboard.py --server.port 8503 --server.headless true
# video/
node scripts/capture.mjs capture/plan.json
ffmpeg -i capture/demo.webm -vf "fps=30,format=yuv420p" -c:v libx264 -crf 14 public/demo.mp4
```

The plan opens a 1280 × 1960 viewport, so the whole customer view fits without scrolling. Scrolling inside
headless Chrome produced torn frames, so the `Demo` scene pans over the tall recording instead. It shows the
customer view for about 8.5 s, then clicks the **Similar items** tab, which opens on its default article. If
you change the plan, update `TRIM`, `TAB_AT` and `PAN` in `src/scenes/Demo.tsx`; the tab switch is at
10.33 s in the current file (`ffmpeg -i public/demo.mp4 -vf "select='gt(scene,0.08)',showinfo" -f null -`).

## Music and sound

Both are synthesized from the cues in `scenes.json`, so there is nothing to license:

```bash
.venv/Scripts/python video/scripts/music.py --root F --mode major --bpm 104   # -> public/music.mp3, public/sfx.mp3
```

## Data rules

`public/data/` (the derived JSON and thumbnails), `public/demo.mp4` and `capture/` are all derived from the
competition data, so they are git-ignored. `extract.py` copies at most 16 product photos; this customer
uses 11. The customer is shown only as an 8-character hash prefix plus the age band that the popularity
source uses.
