import React from "react";
import { AbsoluteFill, OffthreadVideo, interpolate, staticFile, useCurrentFrame } from "remotion";
import { C, FONT } from "../kit/theme";
import { BrowserFrame, Scene, clamp, ease, enter, useLayout } from "../kit/ui";
import { hasFile } from "../kit/Timeline";
import { Head, Source } from "./common";

// public/demo.mp4: the real Streamlit demo (app/dashboard.py) recorded by scripts/capture.mjs from
// capture/plan.json at a 1280 x 1960 viewport, 1.25x -> 1600 x 2450 px. The page is tall so nothing scrolls
// in the browser; the "camera" pans over the recording here instead. The Similar-items tab opens at 10.33 s.
const SRC_W = 1600;
const SRC_H = 2450;
const TRIM = 30; // skip the first second (static)
const TAB_AT = 310 - TRIM; // scene frame where the tab switch happens
const WIN = 0.62; // visible window height / width

// camera keyframes: [scene frame, top of the window in source px]
const PAN: [number, number][] = [
  [0, 0],
  [55, 0],
  [105, 560],
  [160, 560],
  [205, 1180],
  [TAB_AT - 18, 1180],
  [TAB_AT, 0],
  [TAB_AT + 10, 0],
  [TAB_AT + 40, 300],
];
const NOTES = [
  { from: 0, to: 160, k: "Customer view", t: "Recent purchases next to next week's 12 picks." },
  { from: 160, to: TAB_AT, k: "Reasons", t: "Every card shows which sources suggested it." },
  { from: TAB_AT, to: 999, k: "Similar items", t: "CLIP text-embedding neighbours of any article." },
];

export const Demo: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const fw = wide ? 1180 * u : width - 100 * u;
  const vw = fw - 34 * u; // inside the frame's padding
  const vh = vw * WIN;
  const sc = vw / SRC_W;
  const top = interpolate(f, PAN.map((p) => p[0]), PAN.map((p) => p[1]), { ...clamp, easing: ease });
  const s = enter(f, 0, 30);
  const note = NOTES.find((n) => f >= n.from && f < n.to) ?? NOTES[0];
  const frame = (
    <div style={{ opacity: s, transform: `translateY(${(1 - s) * 40}px)` }}>
      <BrowserFrame url="localhost:8503 · app/dashboard.py (Streamlit)" width={fw}>
        <div style={{ width: vw, height: vh, overflow: "hidden", position: "relative", background: "#fff" }}>
          {hasFile("demo.mp4") ? (
            <OffthreadVideo src={staticFile("demo.mp4")} muted trimBefore={TRIM} style={{ position: "absolute", left: 0, top: -top * sc, width: vw, height: SRC_H * sc }} />
          ) : (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", fontFamily: FONT.mono, color: C.muted }}>public/demo.mp4 missing: see video/README.md</div>
          )}
        </div>
      </BrowserFrame>
    </div>
  );
  const notes = (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 * u, width: wide ? 520 * u : fw }}>
      {NOTES.map((n) => {
        const on = n === note;
        return (
          <div key={n.k} style={{ display: "flex", gap: 16 * u, alignItems: "flex-start", opacity: on ? 1 : 0.38 }}>
            <span style={{ width: 6 * u, alignSelf: "stretch", borderRadius: 6, background: on ? C.gold : C.line, flex: "none" }} />
            <div>
              <div style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: 28 * u, color: C.ink, letterSpacing: "-0.02em" }}>{n.k}</div>
              <div style={{ fontFamily: FONT.body, fontSize: 22 * u, color: C.muted, lineHeight: 1.4 }}>{n.t}</div>
            </div>
          </div>
        );
      })}
      <Source size={14}>a real screen recording · photos were downloaded for demo articles only</Source>
    </div>
  );
  if (wide)
    return (
      <Scene dur={dur}>
        <AbsoluteFill style={{ flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 50 * u, padding: 50 * u }}>
          {frame}
          <div style={{ display: "flex", flexDirection: "column", gap: 40 * u }}>
            <Head eyebrow="The real app" text="See it working." hl="working." align="left" size={62} />
            {notes}
          </div>
        </AbsoluteFill>
      </Scene>
    );
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 24 * u, padding: 35 * u }}>
        <Head eyebrow="The real app" text="See it working." hl="working." size={50} />
        {frame}
        <div style={{ display: "flex", gap: 16 * u, alignItems: "flex-start", width: fw }}>
          <span style={{ width: 6 * u, alignSelf: "stretch", borderRadius: 6, background: C.gold, flex: "none" }} />
          <div>
            <div style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: 27 * u, color: C.ink }}>{note.k}</div>
            <div style={{ fontFamily: FONT.body, fontSize: 21 * u, color: C.muted }}>{note.t}</div>
          </div>
        </div>
      </AbsoluteFill>
    </Scene>
  );
};
