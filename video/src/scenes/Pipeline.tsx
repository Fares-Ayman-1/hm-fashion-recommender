import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { C, FONT, SHADOW } from "../kit/theme";
import { Chip, FadeUp, Panel, Scene, enter, progress, useLayout } from "../kit/ui";
import { FEATURES_IN_FILE, IMPORTANCE, NUM, REASON, pct, type Reason } from "../data";
import { Dot, Head, Source } from "./common";

// Source descriptions follow src/hmrec/candidates/generate.py and configs/default.yaml.
const SOURCES: { r: Reason | "all"; name: string; desc: string }[] = [
  { r: "repurchase", name: "Repurchase", desc: "their own purchases from the last 12 weeks" },
  { r: "popular", name: "Age-band popularity", desc: "last week's best sellers in their age band" },
  { r: "cooccur", name: "Bought together", desc: "bought in the same week as their recent items" },
  { r: "similar", name: "CLIP neighbours", desc: "closest articles in CLIP text-embedding space" },
  { r: "all", name: "All four combined", desc: "one candidate list per customer, with a flag per source" },
];
export const STAGE1_BARS_AT = 34;
export const STAGE1_STEP = 8;
export const STAGE1_GROW_AT = 80; // bars land at 116

/** 4. Stage 1: the four candidate sources and their recall on the validation week. */
export const Stage1: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const w = wide ? 1320 * u : width - 90 * u;
  const max = NUM.recall.all;
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 34 : 26) * u, padding: 44 * u }}>
        <Head eyebrow="Stage 1 · candidate generation" text="Four ways to find candidates." hl="Four" />
        <Panel style={{ width: w, padding: `${(wide ? 30 : 24) * u}px ${(wide ? 40 : 28) * u}px`, boxSizing: "border-box", display: "flex", flexDirection: "column", gap: (wide ? 20 : 17) * u }}>
          {SOURCES.map((s, i) => {
            // rows fade in one by one; the bars then grow together, so their lengths are always comparable
            const at = STAGE1_BARS_AT + i * STAGE1_STEP + (s.r === "all" ? 10 : 0);
            const o = enter(f, at, 18);
            const g = progress(f, STAGE1_GROW_AT, STAGE1_GROW_AT + 36);
            const v = NUM.recall[s.r];
            const hero = s.r === "all";
            return (
              <div key={s.r} style={{ opacity: o, borderTop: hero ? `1px solid ${C.line}` : undefined, paddingTop: hero ? (wide ? 18 : 14) * u : 0 }}>
                <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16 * u }}>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 14 * u, flexWrap: "wrap" }}>
                    <span style={{ alignSelf: "center", display: "flex" }}>{hero ? <span style={{ width: 16 * u, height: 16 * u, borderRadius: 4 * u, background: C.gold, display: "inline-block" }} /> : <Dot r={s.r as Reason} size={16 * u} />}</span>
                    <span style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: (hero ? 30 : 27) * u, color: C.ink, letterSpacing: "-0.02em" }}>{s.name}</span>
                    {(wide || hero) && <span style={{ fontFamily: FONT.body, fontSize: (wide ? 21 : 18) * u, color: C.muted }}>{s.desc}</span>}
                  </div>
                  <span style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: (hero ? 40 : 30) * u, color: C.ink, fontVariantNumeric: "tabular-nums" }}>{pct(v)}</span>
                </div>
                {!wide && !hero && <div style={{ fontFamily: FONT.body, fontSize: 18 * u, color: C.muted, marginLeft: 30 * u }}>{s.desc}</div>}
                <div style={{ height: (hero ? 20 : 13) * u, background: "rgba(12,23,51,0.06)", borderRadius: 99, overflow: "hidden", marginTop: 8 * u }}>
                  <div style={{ width: `${(v / max) * 100 * g}%`, height: "100%", borderRadius: 99, background: hero ? C.gold : REASON[s.r as Reason].color, boxShadow: hero ? SHADOW.goldGlow : undefined }} />
                </div>
              </div>
            );
          })}
        </Panel>
        <FadeUp at={110}>
          <div style={{ fontFamily: FONT.body, fontSize: (wide ? 26 : 23) * u, color: C.ink, textAlign: "center", maxWidth: w }}>
            Recall = the share of the validation week's real purchases that appear among the candidates.
          </div>
          <div style={{ marginTop: 8 * u }}>
            <Source align="center">reports/serving/metrics.json · candidate_recall · week {NUM.validWeek}</Source>
          </div>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

const group = (name: string) => (name.startsWith("u_") ? "user" : name.startsWith("i_") || name.endsWith("_no") || name.endsWith("_code") ? "item" : "user × item");

/** 5. Stage 2: LightGBM LambdaRank, top-8 features by gain. */
export const Stage2: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const w = wide ? 1000 * u : width - 90 * u;
  const max = IMPORTANCE[0].share;
  const intro = (
    <div style={{ display: "flex", flexDirection: "column", gap: 22 * u, alignItems: wide ? "flex-start" : "center", maxWidth: wide ? 640 * u : undefined }}>
      <Head eyebrow="Stage 2 · learning to rank" text="A ranker orders the candidates." hl="ranker" align={wide ? "left" : "center"} size={wide ? 64 : 50} />
      <div style={{ display: "flex", gap: 10 * u, flexWrap: "wrap", justifyContent: wide ? "flex-start" : "center" }}>
        <Chip at={22} size={19 * u} tone="gold">LightGBM LambdaRank</Chip>
        <Chip at={25} size={19 * u}>{`${FEATURES_IN_FILE} features`}</Chip>
        <Chip at={28} size={19 * u}>user · item · user × item</Chip>
      </div>
      {wide && (
        <FadeUp at={34}>
          <div style={{ fontFamily: FONT.body, fontSize: 25 * u, color: C.muted, lineHeight: 1.5 }}>
            Trained to put next week's purchases at the top of each customer's list. Bars show each feature's share of the model's total gain.
          </div>
        </FadeUp>
      )}
    </div>
  );
  const bars = (
    <Panel style={{ width: w, padding: `${(wide ? 28 : 22) * u}px ${(wide ? 34 : 26) * u}px`, boxSizing: "border-box", display: "flex", flexDirection: "column", gap: (wide ? 15 : 12) * u }}>
      <div style={{ fontFamily: FONT.mono, fontSize: 15 * u, letterSpacing: "0.16em", textTransform: "uppercase", color: C.muted }}>Top 8 features · share of total gain</div>
      {IMPORTANCE.map((r, i) => {
        const at = 30 + i * 6;
        const g = progress(f, at, at + 34);
        return (
          <div key={r.name} style={{ opacity: Math.min(1, g * 2) }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 * u }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 12 * u, minWidth: 0 }}>
                <span style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: (wide ? 24 : 22) * u, color: C.ink, whiteSpace: "nowrap" }}>{r.label}</span>
                <span style={{ fontFamily: FONT.mono, fontSize: (wide ? 14 : 13) * u, color: C.muted, whiteSpace: "nowrap" }}>{r.name}</span>
              </div>
              <span style={{ display: "flex", alignItems: "baseline", gap: 12 * u }}>
                <span style={{ fontFamily: FONT.mono, fontSize: 13 * u, color: C.muted, whiteSpace: "nowrap" }}>{group(r.name)}</span>
                <span style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: (wide ? 25 : 23) * u, color: C.ink, fontVariantNumeric: "tabular-nums", minWidth: 70 * u, textAlign: "right" }}>{pct(r.share)}</span>
              </span>
            </div>
            <div style={{ height: 11 * u, background: "rgba(12,23,51,0.06)", borderRadius: 99, overflow: "hidden", marginTop: 6 * u }}>
              <div style={{ width: `${(r.share / max) * 100 * g}%`, height: "100%", borderRadius: 99, background: i === 0 ? C.gold : "rgba(12,23,51,0.55)" }} />
            </div>
          </div>
        );
      })}
      <div style={{ opacity: enter(f, 90) }}>
        <Source>reports/feature_importance.json (LightGBM gain)</Source>
      </div>
    </Panel>
  );
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", padding: 44 * u }}>
        <div style={{ display: "flex", flexDirection: wide ? "row" : "column", alignItems: "center", gap: (wide ? 60 : 26) * u }}>
          {intro}
          {bars}
        </div>
      </AbsoluteFill>
    </Scene>
  );
};
