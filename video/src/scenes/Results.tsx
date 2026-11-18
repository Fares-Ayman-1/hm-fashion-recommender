import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { C, FONT, SHADOW } from "../kit/theme";
import { BarList, Callout, FadeUp, Panel, Scene, enter, progress, useLayout } from "../kit/ui";
import { DEMO, NUM, README_NUM, REASON, pct } from "../data";
import { Card, Head, ReasonChip, Source, Thumb } from "./common";

export const RESULTS_BARS_AT = 40;

/** 6. Results: MAP@12 on the held-out week vs two baselines. */
export const Results: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide, width } = useLayout();
  const w = wide ? 1180 * u : width - 100 * u;
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 34 : 28) * u, padding: 44 * u }}>
        <Head eyebrow={`Held-out week ${NUM.validWeek} · ${README_NUM.validDates} · ${README_NUM.validBuyers.toLocaleString("en-US")} buyers`} text={`${pct(NUM.lift)} better than the best baseline.`} hl={pct(NUM.lift)} />
        <Panel style={{ padding: `${32 * u}px ${38 * u}px`, width: w, boxSizing: "border-box" }}>
          <div style={{ fontFamily: FONT.mono, fontSize: 15 * u, letterSpacing: "0.16em", textTransform: "uppercase", color: C.muted, marginBottom: 20 * u }}>MAP@12 · higher is better</div>
          {/* one BarList per row, all starting together: mid-animation the bars keep their true proportions */}
          <div style={{ display: "flex", flexDirection: "column", gap: (wide ? 24 : 21) * u }}>
            {[
              { label: "Global top-12 of last week", value: NUM.popular, display: NUM.popular.toFixed(5) },
              { label: "Repurchase, then age-band popular", value: NUM.repPop, display: NUM.repPop.toFixed(5) },
              { label: "Two-stage ranker", value: NUM.ranker, display: NUM.ranker.toFixed(5), hero: true },
            ].map((row, i) => (
              <div key={row.label} style={{ opacity: enter(f, RESULTS_BARS_AT - 12 + i * 4, 16) }}>
                <BarList at={RESULTS_BARS_AT} width={w - 76 * u} size={(wide ? 27 : 24) * u} max={NUM.ranker} rows={[row]} />
              </div>
            ))}
          </div>
        </Panel>
        <Callout at={100} big={`+${pct(NUM.lift)}`} width={w}>
          MAP@12 asks: of the 12 picks, how many did the customer really buy that week, and how close to the top were they?
        </Callout>
        <FadeUp at={110}>
          <Source align="center">reports/serving/metrics.json · map12 · MLflow run {NUM.runId.slice(0, 8)}…</Source>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

const Compare: React.FC<{ at: number; title: string; before: string; after: string; fracBefore: number; fracAfter: number; note: string }> = ({ at, title, before, after, fracBefore, fracAfter, note }) => {
  const f = useCurrentFrame();
  const { u, wide } = useLayout();
  const g = progress(f, at + 10, at + 46);
  const bar = (label: string, frac: number, value: string, hero: boolean) => (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 * u }}>
        <span style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: 21 * u, color: hero ? C.ink : C.muted }}>{label}</span>
        <span style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: (hero ? 36 : 32) * u, color: C.ink, fontVariantNumeric: "tabular-nums" }}>{value}</span>
      </div>
      <div style={{ height: 14 * u, background: "rgba(12,23,51,0.06)", borderRadius: 99, overflow: "hidden" }}>
        <div style={{ width: `${frac * 100 * g}%`, height: "100%", borderRadius: 99, background: hero ? C.gold : "rgba(12,23,51,0.35)" }} />
      </div>
    </div>
  );
  return (
    <FadeUp at={at}>
      <Panel style={{ width: (wide ? 640 : 960) * u, padding: `${26 * u}px ${30 * u}px`, boxSizing: "border-box", display: "flex", flexDirection: "column", gap: 14 * u }}>
        <div style={{ fontFamily: FONT.mono, fontSize: 15 * u, letterSpacing: "0.16em", textTransform: "uppercase", color: C.muted }}>{title}</div>
        {bar("Without CLIP", fracBefore, before, false)}
        {bar("With CLIP", fracAfter, after, true)}
        <div style={{ fontFamily: FONT.body, fontSize: 20 * u, color: C.ink, lineHeight: 1.4 }}>{note}</div>
      </Panel>
    </FadeUp>
  );
};

/** 7. An honest note: the CLIP source adds little (README numbers, cited). */
export const Honest: React.FC<{ dur: number }> = ({ dur }) => {
  const { u, wide } = useLayout();
  const gain = NUM.ranker - README_NUM.mapNoClip;
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 36 : 22) * u, padding: 44 * u }}>
        <Head eyebrow="An honest note" text="CLIP adds very little here." hl="little" />
        <div style={{ display: "flex", flexDirection: wide ? "row" : "column", gap: (wide ? 28 : 16) * u }}>
          <Compare at={18} title="Candidate recall" before={pct(README_NUM.recallNoClip)} after={pct(NUM.recall.all)} fracBefore={README_NUM.recallNoClip / NUM.recall.all} fracAfter={1} note={`On its own CLIP finds ${pct(NUM.recall.similar)} of purchases, but it lifts the combined recall by only half a point.`} />
          <Compare at={30} title="MAP@12" before={README_NUM.mapNoClip.toFixed(5)} after={NUM.ranker.toFixed(5)} fracBefore={README_NUM.mapNoClip / NUM.ranker} fracAfter={1} note={`+${gain.toFixed(4)}: inside LightGBM's run-to-run noise (about ±${README_NUM.noise}).`} />
        </div>
        <FadeUp at={70}>
          <div style={{ fontFamily: FONT.body, fontSize: (wide ? 24 : 21) * u, color: C.muted, textAlign: "center", maxWidth: 1100 * u }}>
            The embeddings are text-only: the 28.7 GB photo archive was impractical to download here.
          </div>
          <div style={{ marginTop: 8 * u }}>
            <Source align="center">without-CLIP numbers: README.md (MLflow run a03f40ed…) · with CLIP: metrics.json</Source>
          </div>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

/** 8. Explainable: every recommendation carries its reasons (the API's shape, real demo-customer rows). */
export const Explain: React.FC<{ dur: number }> = ({ dur }) => {
  const f = useCurrentFrame();
  const { u, wide } = useLayout();
  const picks = [3, 6, 1].map((rank) => DEMO.recs.find((r) => r.rank === rank)!).filter(Boolean);
  const ex = picks[0];
  const tw = (wide ? 92 : 84) * u;
  const rows = (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 * u, width: (wide ? 820 : 960) * u }}>
      {picks.map((it, i) => (
        <FadeUp key={it.id} at={16 + i * 10}>
          <Panel style={{ padding: 16 * u, display: "flex", gap: 20 * u, alignItems: "center" }}>
            <Card style={{ padding: 5 * u, flex: "none" }}>
              <Thumb it={it} w={tw} radius={8 * u} />
            </Card>
            <div style={{ display: "flex", flexDirection: "column", gap: 9 * u, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 12 * u }}>
                <span style={{ fontFamily: FONT.mono, fontWeight: 500, fontSize: 18 * u, color: C.muted }}>#{it.rank}</span>
                <span style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: 27 * u, color: C.ink, letterSpacing: "-0.02em" }}>{it.name}</span>
                <span style={{ fontFamily: FONT.body, fontSize: 19 * u, color: C.muted, whiteSpace: "nowrap" }}>
                  {it.type} · {it.colour}
                </span>
              </div>
              <div style={{ display: "flex", gap: 8 * u, flexWrap: "wrap" }}>
                {it.reasons.map((r, j) => (
                  <ReasonChip key={r} r={r} size={17 * u} at={30 + i * 10 + j * 3} />
                ))}
              </div>
              <div style={{ fontFamily: FONT.body, fontSize: 19 * u, color: C.ink, lineHeight: 1.35, opacity: enter(f, 44 + i * 10) }}>“{it.reasons.map((r) => REASON[r].text).join("; ")}”</div>
            </div>
          </Panel>
        </FadeUp>
      ))}
    </div>
  );
  const json = [
    `GET /v1/customers/{id}/recommendations`,
    ``,
    `{ "rank": ${ex.rank},`,
    `  "article": {`,
    `    "article_id": ${Number(ex.id)},`,
    `    "name": "${ex.name}",`,
    `    "colour": "${ex.colour}" },`,
    `  "reasons": [`,
    ...ex.reasons.map((r, i) => `    "${r}"${i < ex.reasons.length - 1 ? "," : ""}`),
    `  ],`,
    `  "explanation": "…" }`,
  ];
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 34 : 24) * u, padding: 44 * u }}>
        <Head eyebrow={`Explainable · demo customer ${DEMO.customer.short}…`} text="Every pick says why." hl="why." />
        <div style={{ display: "flex", gap: 28 * u, alignItems: "center" }}>
          {rows}
          {wide && (
            <FadeUp at={40}>
              <Panel style={{ padding: `${22 * u}px ${26 * u}px`, background: C.card2, boxShadow: SHADOW.card }}>
                {json.map((l, i) => (
                  <div key={i} style={{ fontFamily: FONT.mono, fontSize: (i === 0 ? 15 : 18) * u, color: i === 0 ? C.muted : C.ink, whiteSpace: "pre", lineHeight: 1.5, opacity: enter(f, 46 + i * 2, 14) }}>
                    {l || " "}
                  </div>
                ))}
              </Panel>
            </FadeUp>
          )}
        </div>
        <FadeUp at={70}>
          <Source align="center">{wide ? "reports/serving/recommendations.parquet · reasons · same shape in the FastAPI service" : "GET /v1/customers/{id}/recommendations · reasons + explanation"}</Source>
        </FadeUp>
      </AbsoluteFill>
    </Scene>
  );
};

/** 10. Kaggle score and engineering. */
export const Engineering: React.FC<{ dur: number }> = ({ dur }) => {
  const { u, wide } = useLayout();
  const mon = NUM.mon;
  const tiles = [
    { k: "FastAPI", t: "/v1 service: picks with reasons and a plain-English explanation" },
    { k: "Batch scoring", t: `${NUM.scored.toLocaleString("en-US")} customers scored` },
    { k: "Docker", t: "API + demo with docker compose" },
    { k: "CI", t: "GitHub Actions: ruff, mypy, pytest" },
    { k: "MLflow", t: "every run tracked; RESULTS.md generated from it" },
    {
      k: "Monitoring",
      t: `week ${NUM.validWeek}: ${NUM.monAlerts} of 3 drift alerts`,
      m: `new items ${pct(mon.new_article_share)} · top-100 overlap ${mon.top100_overlap.toFixed(2)} · price PSI ${mon.price_psi.toFixed(3)}`,
    },
  ];
  const kaggle = (
    <FadeUp at={16}>
      <Panel float style={{ padding: `${28 * u}px ${32 * u}px`, width: (wide ? 520 : 960) * u, boxSizing: "border-box", borderTop: `6px solid ${C.gold}` }}>
        <div style={{ fontFamily: FONT.mono, fontSize: 15 * u, letterSpacing: "0.16em", textTransform: "uppercase", color: C.muted }}>Kaggle · hidden test week · MAP@12</div>
        <div style={{ display: "flex", flexDirection: wide ? "column" : "row", gap: (wide ? 14 : 60) * u, marginTop: 16 * u }}>
          {[
            { v: NUM.kagglePrivate, l: "private leaderboard" },
            { v: NUM.kagglePublic, l: "public leaderboard" },
          ].map((s) => (
            <div key={s.l}>
              <div style={{ fontFamily: FONT.head, fontWeight: 700, fontSize: (wide ? 76 : 66) * u, color: C.ink, letterSpacing: "-0.04em", lineHeight: 1 }}>{s.v.toFixed(5)}</div>
              <div style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: 22 * u, color: C.muted, marginTop: 4 * u }}>{s.l}</div>
            </div>
          ))}
        </div>
        <div style={{ fontFamily: FONT.body, fontSize: 19 * u, color: C.ink, marginTop: 16 * u, lineHeight: 1.4 }}>A late submission. Lower than validation: the test week is a different, unseen week.</div>
        <div style={{ marginTop: 8 * u }}>
          <Source size={14}>reports/kaggle_score.json</Source>
        </div>
      </Panel>
    </FadeUp>
  );
  const grid = (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(2, ${(wide ? 400 : 472) * u}px)`, gap: 14 * u }}>
      {tiles.map((t, i) => (
        <FadeUp key={t.k} at={30 + i * 5}>
          <Panel style={{ padding: `${(wide ? 20 : 16) * u}px ${22 * u}px`, height: "100%", boxSizing: "border-box", display: "flex", flexDirection: "column", gap: 6 * u }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 * u }}>
              <span style={{ width: 10 * u, height: 10 * u, borderRadius: 10, background: t.k === "Monitoring" ? C.green : C.gold }} />
              <span style={{ fontFamily: FONT.head, fontWeight: 600, fontSize: 25 * u, color: C.ink, letterSpacing: "-0.02em" }}>{t.k}</span>
            </div>
            <div style={{ fontFamily: FONT.body, fontSize: (wide ? 19 : 18) * u, color: C.muted, lineHeight: 1.35 }}>{t.t}</div>
            {t.m && <div style={{ fontFamily: FONT.mono, fontSize: 13 * u, color: C.ink, lineHeight: 1.4 }}>{t.m}</div>}
          </Panel>
        </FadeUp>
      ))}
    </div>
  );
  return (
    <Scene dur={dur}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: (wide ? 36 : 22) * u, padding: 44 * u }}>
        <Head eyebrow="Leaderboard and engineering" text="Scored on Kaggle. Built to run." hl="run." size={wide ? 66 : 50} />
        <div style={{ display: "flex", flexDirection: wide ? "row" : "column", gap: (wide ? 30 : 16) * u, alignItems: "stretch" }}>
          {kaggle}
          {grid}
        </div>
      </AbsoluteFill>
    </Scene>
  );
};
