# Score model decision (fixed before the test set) — 2026-10-03

Data: train = 80 AVI training participants (avi_train1 + avi_train2), q1+q2;
dev = 30 AVI validation participants (avi_dev_subset). Target: hireability.

| model      | features | CV rho (train) | dev rho | dev 95% CI     |
|------------|----------|----------------|---------|----------------|
| rule       | 0        | –              | -0.09   | [-0.45, 0.28]  |
| length     | 3        | 0.48           | 0.28    | [-0.12, 0.61]  |
| audio_text | 17       | 0.50           | 0.27    | [-0.09, 0.56]  |
| visual     | 27       | -0.24          | 0.16    | [-0.20, 0.53]  |
| all        | 44       | 0.36           | 0.27    | [-0.08, 0.55]  |

Decision
- System score = **audio_text** (ridge). Learned models tie on dev; audio_text has
  the best training CV and is the base for the RQ2 comparison.
- Reported on the test set (run once, `train_score_model.py --final`):
  1. audio_text vs length   — do speech features add beyond answer length?
  2. all vs audio_text      — RQ2: do visual features add?
  3. audio_text vs rule     — learned scoring vs hand-set bands
- No further tuning after this point.

Observations (dev, exploratory)
- Hand-set bands do not agree with recruiters (rho -0.09 here; -0.17 in the first dev run).
- Rating is driven mostly by answer length / amount of speech.
- Visual features alone are weak and lower training CV when added.
- Ridge weights are all small (|w| < 0.05): weak, spread-out signal — do not
  interpret single features.

## Approach 1b (added 2026-10-04)

The audio_text model above is now also used as a scoring approach:
`approaches/approach_1b_learned/` (train.py saves it, score.py applies it to any clip).
- System model = trained on train+dev (same as `--final`), so the test set is still untouched.
- Overall score shown to users = percentile among AVI training participants (0-100) + band.
- Dimension coaching still uses Approach 1 bands.
- Dev check with fit=train: ρ = 0.27, identical to the table above (linear model, so the
  mean of clip predictions equals the prediction of the mean features).
- Watch on test: gender SMD of the 1b score on dev was -0.59 vs -0.25 for recruiters (n=28).

## Protocol v2 — research pivot (written 2026-10-04, before any M3 result and before the test set)

The study now compares scoring METHODS (roadmap RQ1/RQ2), so the rule is:
- Every method below is reported on the test set, whatever its dev result. No method is
  dropped or picked because of dev numbers (avoids choosing the luckiest of many on n=30).
  - M0 length, M1 rule, M2 audio_text / visual / all (ridge on Tier-1 features)
  - M3 text_emb (pretrained transcript embedding, `--text-emb minilm`; one encoder, fixed now)
  - M4 text_emb+audio_text, text_emb+all
  - M5 local LLM zero-shot (if run)
- Dev is used only to check that each method runs sensibly and to fix hyper-parameter grids.
- Primary comparison (fixed now): M3 vs M0, M4 vs M2 (audio_text), best-by-design system
  = M4 text_emb+audio_text. Paired bootstrap Δρ with 95% CI.
- Tier-1 visual features were re-measured after the aspect-ratio fix (roadmap 7.6);
  the dev table above (visual / all) predates that fix and will be re-run.
- Encoder for M3: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0). Others
  (bge-small, e5-base) only as a reported sensitivity check, not to pick the best.

## Data expansion (written 2026-10-04, before any test-set processing or result)

Two tracks, fixed now:
- **Audio/text track** — every AVI participant, official subject-level split:
  train 452 / dev 64 / test 130 (q1 + q2). `run_speech_pipeline.py` (audio, prosody,
  WhisperX transcript, speech features, text embedding; no video frames).
  Methods: M0 length, M2 audio_text, M3 text_emb, M4 text_emb+audio_text, (M5 local LLM).
  `train_score_model.py --data full --text-emb minilm`
- **Visual track** — subset with frames/landmarks (disk limit: ~30-40 GB for all):
  train 80 / dev 30 / test 50 (avi_eval_subset, a subset of the 130 test people).
  Methods: M1 rule, M2 visual / all, M4 text_emb+all; RQ2 comparisons on the same 50.
- Learning curve (n_train = 25, 50, 100, 200, 300, all; 5 random repeats) reported for
  length, audio_text, text_emb, text_emb+audio_text — answers "does more data help?".
- The test split (130) is processed by the pipeline but no model is evaluated on it until
  the single `--final` run per track.
- Equal-data comparison (added before any result): the visual-track run also includes the
  audio/text methods (`train_score_model.py --text-emb minilm`), so ALL methods are compared
  on the same 80 training / 50 test people. RQ2 is answered there. The full-data run shows
  the best achievable audio/text result; methods are never compared ACROSS the two runs.

## Data decision v3 — ONE equal dataset (written 2026-10-04 18:35, still before any result
## of M3/M4 and before the test set; supersedes "Data expansion" above)

All methods use the same people:
- train 100 = avi_train1 (40) + avi_train2 (40) + avi_train3 (20, stratified, new)
- dev 30 = avi_dev_subset (unchanged)
- test 50 = avi_eval_subset (Testing split, stratified, not yet processed)
- q1 + q2, full Tier 1 (video + audio) for everyone, so text, audio and visual methods are
  compared on identical data. Command: `train_score_model.py --text-emb minilm --learning-curve`
- Learning curve inside the 100 (25 / 50 / 100) shows whether more data would help.
- `run_speech_pipeline.py` / `--data full` remain available as an optional extra analysis
  only; not part of the main results.

## M5 settings (fixed 2026-10-04, before any M5 output)
- Local Ollama only (script refuses non-local hosts); model qwen2.5:3b (fits 4 GB VRAM)
- Prompt m5-v1: question + "strong answer" focus (delivery not judged; transcript only),
  one-sentence reason, then rating 1-10; temperature 0, seed 42, num_ctx 4096
- Reported two ways: llm_zero_shot (the rating itself, no training) and llm+audio_text
  (rating as one extra ridge feature). Other local models only as a sensitivity check.

## Dev check v3 (2026-10-04 22:09) — train 100 / dev 30, after the aspect-ratio fix
Checking only; nothing is tuned or dropped because of this (protocol v2).
Full table: report/score_model_dev_minilm_llm-qwen2.5_3b.csv

| model | dev rho | 95% CI | train CV rho |
|---|---|---|---|
| rule (M1) | -0.00 | [-0.34, 0.33] | - |
| llm_zero_shot (M5) | 0.33 | [-0.02, 0.62] | - |
| length (M0) | 0.28 | [-0.12, 0.61] | 0.40 |
| audio_text (M2) | 0.28 | [-0.05, 0.56] | 0.32 |
| visual (M2) | 0.29 | [-0.11, 0.65] | -0.03 |
| all (M2) | 0.27 | [-0.13, 0.59] | 0.29 |
| text_emb (M3) | 0.25 | [-0.11, 0.56] | 0.25 |
| text_emb+audio_text (M4) | 0.25 | [-0.11, 0.54] | 0.27 |
| text_emb+all (M4) | 0.29 | [-0.06, 0.58] | 0.26 |
| llm+audio_text | 0.32 | [-0.01, 0.59] | 0.39 |

Notes: no paired difference is distinguishable on n=30 (all CIs include 0). Learning curve:
only text_emb rises steadily with more training people (0.15 -> 0.22 -> 0.25 at 25/50/100);
length is flat (~0.28). The 'all' model picked a small alpha (0.56) with large opposite-sign
weights on correlated pause features — a sign of instability, not interpretable.

