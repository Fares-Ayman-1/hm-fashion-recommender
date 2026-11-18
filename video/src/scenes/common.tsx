import React from "react";
import { Img, useCurrentFrame } from "remotion";
import { C, FONT, SHADOW } from "../kit/theme";
import { Eyebrow, Words, enter, useLayout } from "../kit/ui";
import { REASON, img, type Item, type Reason } from "../data";

/** Eyebrow + kinetic headline, sized for wide or square. */
export const Head: React.FC<{ eyebrow: string; text: string; hl: string; at?: number; size?: number; align?: "center" | "left" }> = ({ eyebrow, text, hl, at = 0, size, align = "center" }) => {
  const { u, wide, width } = useLayout();
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: align === "center" ? "center" : "flex-start", gap: 20 * u }}>
      <Eyebrow at={at} size={17 * u}>
        {eyebrow}
      </Eyebrow>
      <Words text={text} at={at + 4} size={(size ?? (wide ? 70 : 56)) * u} highlight={[hl]} align={align} maxWidth={width - 120 * u} />
    </div>
  );
};

/** A coloured dot for a candidate source (fills only, never coloured text). */
export const Dot: React.FC<{ r: Reason; size: number }> = ({ r, size }) => (
  <span style={{ width: size, height: size, borderRadius: size, background: REASON[r].color, display: "inline-block", flex: "none", border: r === "repurchase" ? "none" : `1px solid rgba(12,23,51,0.12)` }} />
);

/** Reason chip: coloured dot + navy label. */
export const ReasonChip: React.FC<{ r: Reason; size: number; at?: number }> = ({ r, size, at = 0 }) => {
  const f = useCurrentFrame();
  const s = enter(f, at, 18);
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: size * 0.45, fontFamily: FONT.body, fontWeight: 600, fontSize: size, color: C.ink, background: C.card2, border: `1px solid ${C.line}`, borderRadius: 999, padding: `${size * 0.25}px ${size * 0.7}px`, whiteSpace: "nowrap", opacity: s, transform: `translateY(${(1 - s) * 8}px)` }}>
      <Dot r={r} size={size * 0.62} />
      {REASON[r].label}
    </span>
  );
};

/** Product thumbnail (3:4) or a neutral "no photo" tile. */
export const Thumb: React.FC<{ it: Item; w: number; radius?: number }> = ({ it, w, radius }) => {
  const { u } = useLayout();
  const src = img(it);
  const h = (w * 4) / 3;
  const r = radius ?? 12 * u;
  if (src) return <Img src={src} style={{ width: w, height: h, objectFit: "cover", borderRadius: r, display: "block", background: "#EEECE8" }} />;
  return (
    <div style={{ width: w, height: h, borderRadius: r, background: C.card2, border: `1px dashed ${C.line}`, boxSizing: "border-box", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 6 * u, textAlign: "center", padding: 8 * u }}>
      <span style={{ fontFamily: FONT.body, fontWeight: 600, fontSize: Math.max(13 * u, w * 0.11), color: C.ink }}>{it.type}</span>
      <span style={{ fontFamily: FONT.mono, fontSize: Math.max(10 * u, w * 0.075), color: C.muted, letterSpacing: "0.1em" }}>NO PHOTO</span>
    </div>
  );
};

/** Small white card holding a thumbnail. */
export const Card: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => {
  const { u } = useLayout();
  return <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 16 * u, padding: 8 * u, boxShadow: SHADOW.card, ...style }}>{children}</div>;
};

/** Legend of the four sources. */
export const Legend: React.FC<{ at?: number; size: number; only?: Reason[] }> = ({ at = 0, size, only }) => {
  const { u } = useLayout();
  const rs = only ?? (["repurchase", "popular", "cooccur", "similar"] as Reason[]);
  return (
    <div style={{ display: "flex", gap: 12 * u, flexWrap: "wrap", justifyContent: "center" }}>
      {rs.map((r, i) => (
        <ReasonChip key={r} r={r} size={size} at={at + i * 3} />
      ))}
    </div>
  );
};

/** Small mono source line under a number ("RESULTS.md", "reports/..."). */
export const Source: React.FC<{ children: React.ReactNode; size?: number; align?: "left" | "center" }> = ({ children, size = 15, align }) => {
  const { u } = useLayout();
  return <div style={{ fontFamily: FONT.mono, fontSize: size * u, color: C.muted, letterSpacing: "0.04em", textAlign: align }}>{children}</div>;
};
