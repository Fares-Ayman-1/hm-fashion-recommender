import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { C, FONT, SHADOW } from "../kit/theme";
import { FadeUp, Panel, Scene, enter, progress, useLayout } from "../kit/ui";
import { DEMO, NUM, README_NUM } from "../data";
import { Card, Dot, Head, Legend, Source, Thumb } from "./common";

const Label: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { u } = useLayout();
  return <div style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: 15 * u, letterSpacing: "0.16em", textTransform: "uppercase", color: C.muted, marginBottom: 12 * u }}>{children}</div>;
};

const Arrow: React.FC<{ at: number; down?: boolean }> = ({ at, down }) => {
  const f = useCurrentFrame();
  const { u } = useLayout();
  const p = progress(f, at, at + 22);
  const len = (down ? 34 : 64) * u;
  return (
    <svg width={down ? 24 * u : len + 16 * u} height={down ? len + 16 * u : 24 * u} style={{ flex: "none", overflow: "visible" }}>
      {down ? (
        <>
          <line x1={12 * u} y1={0} x2={12 * u} y2={len * p} stroke={C.gold} strokeWidth={4 * u} strokeLinecap="round" />
          <path d={`M ${4 * u} ${len - 4 * u} L ${12 * u} ${len + 6 * u} L ${20 * u} ${len - 4 * u}`} fill="none" stroke={C.gold} strokeWidth={4 * u} strokeLinecap="round" strokeLinejoin="round" opacity={p > 0.9 ? 1 : 0} />
        </>
      ) : (
        <>
          <line x1={0} y1={12 * u} x2={len * p} y2={12 * u} stroke={C.gold} strokeWidth={4 * u} strokeLinecap="round" />
          <path d={`M ${len - 4 * u} ${4 * u} L ${len + 6 * u} ${12 * u} L ${len - 4 * u} ${20 * u}`} fill="none" stroke={C.gold} strokeWidth={4 * u} strokeLinecap="round" strokeLinejoin="round" opacity={p > 0.9 ? 1 : 0} />
        </>
      )}
    </svg>
  );
};

/** 1. Hook: one real demo customer's recent purchases become 12 recommendations, each with its reasons. */
export const Hook: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide } = useLayout();
  const pw = (wide ? 128 : 92) * u; // purchase thumb width
  const rw = (wide ? 124 : 112) * u; // recommendation thumb width
  const gap = (wide ? 12 : 9) * u;
  const past = (
    <Panel style={{ padding: 22 * u }}>
      <Label>Recent purchases</Label>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${wide ? 2 : 4}, auto)`, gap }}>
        {DEMO.past.map((it, i) => {
          const s = enter(f, 8 + i * 3);
          return (
            <Card key={it.id} style={{ opacity: s, transform: `translateY(${(1 - s) * 20}px)` }}>
              <Thumb it={it} w={pw} radius={10 * u} />
              <div style={{ fontFamily: FONT.mono, fontSize: 12 * u, color: C.muted, marginTop: 6 * u, textAlign: "center" }}>{it.date}</div>
            </Card>
          );
        })}
      </div>
    </Panel>
  );
  const recs = (
    <Panel style={{ padding: 22 * u }}>
      <Label>Recommended for next week · top {NUM.k}</Label>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, auto)", gap }}>
        {DEMO.recs.map((it, i) => {
          const s = enter(f, 52 + i * 3);
          return (
            <Card key={it.id + i} style={{ opacity: s, transform: `translateY(${(1 - s) * 20}px) scale(${0.96 + 0.04 * s})` }}>
              <Thumb it={it} w={rw} radius={10 * u} />
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 7 * u, height: 16 * u }}>
                <span style={{ fontFamily: FONT.mono, fontSize: 13 * u, color: C.ink, fontWeight: 500 }}>#{it.rank}</span>
                <span style={{ display: "flex", gap: 4 * u }}>
                  {it.reasons.map((r, j) => (
                    <span key={r} style={{ opacity: enter(f, 92 + i * 2 + j * 2, 14) }}>
                      <Dot r={r} size={12 * u} />
                    </span>
                  ))}
                </span>
              </div>
            </Card>
          );
        })}
      </div>
    </Panel>
  );
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 34 : 18) * u, padding: `${40 * u}px ${30 * u}px` }}>
        <Head eyebrow={`Demo customer ${DEMO.customer.short}… · age band ${DEMO.customer.band}`} text="Recent purchases in. Twelve picks out." hl="Twelve" size={wide ? 68 : 50} />
        <div style={{ display: "flex", flexDirection: wide ? "row" : "column", alignItems: "center", gap: (wide ? 18 : 6) * u }}>
          {past}
          <Arrow at={38} down={!wide} />
          {recs}
        </div>
        <FadeUp at={112}>
          <Legend size={(wide ? 19 : 17) * u} at={112} />
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

/** 2. Scale: dataset size, Wednesday-to-Tuesday weeks, the leakage test. */
export const Scale: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide } = useLayout();
  const stats = [
    { big: `${(NUM.customers / 1e6).toFixed(2)}M`, exact: NUM.customers.toLocaleString("en-US"), what: "customers" },
    { big: `${(NUM.transactions / 1e6).toFixed(1)}M`, exact: NUM.transactions.toLocaleString("en-US"), what: "purchases" },
    { big: `${Math.floor(NUM.articles / 1000)}k`, exact: NUM.articles.toLocaleString("en-US"), what: "articles" },
  ];
  const cardW = (wide ? 400 : 312) * u;
  const days = ["W", "T", "F", "S", "S", "M", "T"];
  const weeks = [99, 100, 101, 102, 103, 104, 105];
  const wk = (w: number) =>
    w === NUM.testWeek ? { fill: "transparent", border: `2px dashed ${C.ink}`, tag: "test" } : w === NUM.validWeek ? { fill: C.gold, border: `1px solid ${C.gold}`, tag: "valid" } : w >= 101 ? { fill: "rgba(12,23,51,0.82)", border: "none", tag: "train" } : { fill: "rgba(12,23,51,0.12)", border: "none", tag: "" };
  const cell = (wide ? 74 : 62) * u;
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 40 : 30) * u, padding: 50 * u }}>
        <Head eyebrow="The data · H&M on Kaggle" text="Two years of real purchases." hl="real" />
        <div style={{ display: "flex", gap: (wide ? 28 : 14) * u }}>
          {stats.map((s, i) => (
            <FadeUp key={s.what} at={16 + i * 5}>
              <Panel style={{ width: cardW, padding: `${26 * u}px ${24 * u}px`, boxSizing: "border-box", borderTop: `6px solid ${C.gold}` }}>
                <div style={{ fontFamily: FONT.head, fontWeight: 700, fontSize: (wide ? 96 : 70) * u, color: C.ink, letterSpacing: "-0.04em", lineHeight: 1 }}>{s.big}</div>
                <div style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: (wide ? 30 : 25) * u, color: C.ink, marginTop: 10 * u }}>{s.what}</div>
                <div style={{ fontFamily: FONT.mono, fontSize: (wide ? 17 : 14) * u, color: C.muted, marginTop: 4 * u }}>{s.exact}</div>
              </Panel>
            </FadeUp>
          ))}
        </div>
        <div style={{ display: "flex", flexDirection: wide ? "row" : "column", gap: 22 * u, alignItems: "stretch" }}>
          <FadeUp at={44}>
            <Panel style={{ padding: `${22 * u}px ${26 * u}px`, display: "flex", gap: 26 * u, alignItems: "center" }}>
              <div>
                <Label>Weeks run Wed → Tue</Label>
                <div style={{ display: "flex", gap: 5 * u }}>
                  {days.map((d, i) => (
                    <div key={i} style={{ width: 30 * u, height: 30 * u, borderRadius: 8 * u, background: i === 0 ? C.gold : C.card2, border: `1px solid ${i === 0 ? C.gold : C.line}`, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: FONT.mono, fontSize: 14 * u, color: C.ink, opacity: enter(f, 50 + i * 2, 14) }}>
                      {d}
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ display: "flex", gap: 6 * u, alignItems: "flex-end" }}>
                {weeks.map((w, i) => {
                  const st = wk(w);
                  return (
                    <div key={w} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 * u, opacity: enter(f, 58 + i * 3, 16) }}>
                      <span style={{ fontFamily: FONT.mono, fontSize: 12 * u, color: C.ink, height: 14 * u }}>{st.tag}</span>
                      <div style={{ width: cell * 0.62, height: cell * 0.62, borderRadius: 8 * u, background: st.fill, border: st.border, boxSizing: "border-box" }} />
                      <span style={{ fontFamily: FONT.mono, fontSize: 13 * u, color: C.muted }}>{w}</span>
                    </div>
                  );
                })}
              </div>
            </Panel>
          </FadeUp>
          <FadeUp at={72}>
            <Panel style={{ padding: `${22 * u}px ${26 * u}px`, maxWidth: (wide ? 520 : 1000) * u, height: "100%", boxSizing: "border-box", display: "flex", flexDirection: "column", justifyContent: "center", gap: 8 * u, borderLeft: `6px solid ${C.green}` }}>
              <div style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: 28 * u, color: C.ink, letterSpacing: "-0.02em" }}>No leakage, tested.</div>
              <div style={{ fontFamily: FONT.body, fontSize: 21 * u, color: C.muted, lineHeight: 1.4 }}>Scramble week W and every later week: the features for week W don't change.</div>
              <Source size={14}>tests/test_candidates.py</Source>
            </Panel>
          </FadeUp>
        </div>
      </AbsoluteFill>
    </Scene>
  );
};

/** 3. The funnel: catalogue -> ~182 candidates per customer -> top 12 (not to scale). */
export const Funnel: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const maxW = wide ? 1200 * u : width - 100 * u;
  const rows = [
    { w: 1, fill: "rgba(12,23,51,0.86)", ink: "#FFFFFF", big: NUM.articles.toLocaleString("en-US"), what: "articles in the catalogue", stage: "" },
    { w: 0.56, fill: C.blue, ink: C.ink, big: `≈${Math.round(README_NUM.candidatesPerCustomer)}`, what: "candidates per customer", stage: "Stage 1 · four candidate sources" },
    { w: 0.25, fill: C.gold, ink: C.ink, big: String(NUM.k), what: "recommendations", stage: "Stage 2 · LightGBM LambdaRank" },
  ];
  const h = (wide ? 104 : 96) * u;
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 40 * u, padding: 50 * u }}>
        <Head eyebrow="Two stages, like production recommenders" text={`From ${Math.floor(NUM.articles / 1000)}k articles to 12.`} hl="12." />
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 * u, width: maxW }}>
          {rows.map((r, i) => {
            const at = 20 + i * 26;
            const p = progress(f, at, at + 30);
            return (
              <React.Fragment key={i}>
                {r.stage && (
                  <div style={{ display: "flex", alignItems: "center", gap: 12 * u, opacity: enter(f, at - 6, 18) }}>
                    <svg width={18 * u} height={22 * u}>
                      <path d={`M ${3 * u} ${4 * u} L ${9 * u} ${16 * u} L ${15 * u} ${4 * u}`} fill="none" stroke={C.gold} strokeWidth={3.5 * u} strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <span style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: (wide ? 17 : 15) * u, letterSpacing: "0.14em", textTransform: "uppercase", color: C.ink }}>{r.stage}</span>
                  </div>
                )}
                <div style={{ width: maxW * r.w * (0.85 + 0.15 * p), minWidth: (wide ? 300 : 250) * u, height: h, borderRadius: h / 2, background: r.fill, opacity: p, boxShadow: i === 2 ? SHADOW.goldGlow : SHADOW.card, display: "flex", alignItems: "center", justifyContent: "center", gap: 16 * u, padding: `0 ${26 * u}px`, boxSizing: "border-box" }}>
                  <span style={{ fontFamily: FONT.head, fontWeight: 700, fontSize: (wide ? 50 : 44) * u, color: r.ink, letterSpacing: "-0.03em" }}>{r.big}</span>
                  {(i === 0 || wide) && <span style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: (wide ? 24 : 22) * u, color: r.ink, whiteSpace: "nowrap" }}>{r.what}</span>}
                </div>
                {i > 0 && !wide && <div style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: 22 * u, color: C.ink, opacity: p, marginTop: -4 * u }}>{r.what}</div>}
              </React.Fragment>
            );
          })}
        </div>
        <FadeUp at={100}>
          <Source size={15} align="center">
            not to scale · ≈182 = {README_NUM.candidatesPerCustomer} on average (RESULTS.md) · only articles sold in the last 4 weeks
          </Source>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};
