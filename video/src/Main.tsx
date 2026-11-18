import React from "react";
import scenesJson from "../scenes.json";
import { Timeline, type SceneSpec } from "./kit/Timeline";
import { SeriesOutro, SeriesTitle } from "./kit/series";
import { Funnel, Hook, Scale } from "./scenes/Intro";
import { Stage1, Stage2 } from "./scenes/Pipeline";
import { Engineering, Explain, Honest, Results } from "./scenes/Results";
import { Demo } from "./scenes/Demo";

export const SCENES = scenesJson as SceneSpec[];

// Week number of this project in the series (1-4).
const WEEK = 3;

const registry: Record<string, React.FC<{ dur: number }>> = {
  hook: Hook,
  title: ({ dur }) => <SeriesTitle week={WEEK} dur={dur} />,
  scale: Scale,
  funnel: Funnel,
  stage1: Stage1,
  stage2: Stage2,
  results: Results,
  honest: Honest,
  explain: Explain,
  demo: Demo,
  eng: Engineering,
  outro: ({ dur }) => (
    <SeriesOutro
      week={WEEK}
      dur={dur}
      repo="github.com/Fares-Ayman-1/hm-fashion-recommender"
      kaggle="kaggle.com/code/faresayman1/h-and-m-two-stage-recommender-lambdarank"
      scope="A research project on public competition data, not a production store."
      credit="Data: H&M Personalized Fashion Recommendations (Kaggle), non-commercial use"
    />
  ),
};

export const Main: React.FC = () => <Timeline scenes={SCENES} registry={registry} />;
