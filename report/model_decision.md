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
