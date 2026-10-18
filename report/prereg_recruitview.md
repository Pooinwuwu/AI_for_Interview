# Pre-registration: AVI -> RecruitView cross-dataset test (RQ1 only)

Written 2026-10-17 19:35 (UTC+7), BEFORE any RecruitView audio, transcript or feature was computed
and before any result was seen. Changes after this point are reported as post-hoc.

## Purpose
Check whether the AVI finding "audio/text features predict expert ratings better than answer
length" also holds on a second dataset. Research use only (RecruitView CC BY-NC 4.0 + authors'
policy): no model is trained on RecruitView, nothing from it is used in the web app or coach,
nothing leaves this computer, only aggregate results are reported. No video / face processing.

## Fixed design
- Sample: as drawn on 2026-10-08, input/ground_truth/recruitview_sample.csv
  (300 clips, one per participant, seed 0). Not redrawn.
- Models: M0 length, M2 audio/text, M3 MiniLM text embedding, M4 MiniLM + audio/text.
  Same features and Ridge (alpha grid) as on AVI; fitted on AVI train + val (hireability),
  q1 + q2. AVI test set not used again.
- RecruitView processing: our own pipeline (ffmpeg, Parselmouth, WhisperX); the dataset's
  transcripts are not used.
- Metric: Spearman rho with 95% bootstrap CI (2000 resamples); paired delta rho vs M0.

## Hypotheses
- **H1-RV (primary):** M2 audio/text > M0 length on `interview_score`
  (paired delta rho > 0, 95% CI excludes 0).
- H2-RV (secondary): M3 > M0 on `interview_score`.
- H3-RV (secondary): M4 > M0 on `interview_score`.
- Descriptive only: all models on the other five RecruitView scores; length alone vs each score.

## Expectations stated in advance
Absolute rho will likely be lower than on AVI (clips ~30 s vs 1-2 min, 76 questions vs 2,
different raters and rating method). The question is the ordering (M2 vs M0), not the level.

## Addendum 2026-10-17 19:50 (UTC+7) — still before any RecruitView feature or result
Added a secondary set to separate "the finding does not transfer" from "the AVI-trained model
does not transfer to other question types" (AVI models learn from generic q1/q2 answers only).
- **Intro set:** one self-introduction clip ("Introduce yourself" / "Tell me about yourself")
  per participant -> input/ground_truth/recruitview_intro.csv, 329 clips / 329 participants
  (68 shared with the random sample; otherwise seeded random choice among a person's intro clips).
- **H1-RV-intro (secondary):** M2 audio/text > M0 length on `interview_score` in the intro set
  (paired delta rho, 95% CI). Same models, same fit on AVI, no changes.
- Reading rule fixed in advance:
  - both sets M2 > M0 -> finding transfers;
  - intro yes, random no -> finding likely holds but the model does not transfer to new question types;
  - neither -> finding does not transfer to RecruitView.
- The random 300 set and H1-RV remain the primary test.

## Post-hoc note 2026-10-17 23:55 (UTC+7) — written AFTER first results were seen
First output (random300, interview_score): M0 length rho = -0.52, M2 audio/text -0.42,
M3 -0.26, M4 -0.30; intro set: M0 -0.72, M2 -0.67, M3 -0.50. All AVI-trained models correlate
NEGATIVELY with RecruitView scores.
Diagnostic (not pre-registered): with word count held constant, every acoustic cue points the
opposite way to AVI and to the speech literature (intro set vs confidence_score / AVI vs hireability):
silence ratio +0.35 / -0.30, mean pause +0.26 / -0.21, intensity -0.27 / +0.19, jitter +0.26 / -0.12.
Raw word count vs RecruitView scores: -0.63 to -0.71. The RecruitView paper and README do not
state the sign convention of the MNL scores. Working hypothesis: the distributed scores are
oriented so that higher = less of the attribute, OR the raters strongly preferred short answers.
Cannot be decided from the data alone -> ask the dataset authors.
Consequence for H1-RV as written ("M2 > M0", signed): numerically M2 (-0.42) > M0 (-0.52), but this
is not a meaningful confirmation. Under either sign, |M0| >= |M2|: answer length carries at least as
much signal as audio/text on RecruitView, i.e. the AVI finding is NOT replicated.
