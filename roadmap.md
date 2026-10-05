# Roadmap — Research plan v2 (seminar)

**Last Updated:** 2026-10-04
**Pivot:** web app dropped (dataset licences forbid interview apps). The project is now a
**comparative research study**: which methods score interviews closest to human raters,
which modalities matter, and whether evidence + a checker make AI feedback more truthful.

**Main dataset:** AVI-Personality (local only). RecruitView = evaluation only. Own clips = Gemini feedback study.

---

## 🎯 Research questions

| | Question | Data | Metric |
|---|---|---|---|
| **RQ1** | Which scoring method agrees best with human ratings? | AVI (train/dev/test); RecruitView (cross-dataset test) | Spearman ρ + 95% bootstrap CI, paired Δρ |
| **RQ2** | Which modality carries the signal: text, audio, visual? | same | ablation (drop one group at a time) |
| **RQ3** | Do evidence grounding + an automatic checker reduce hallucination in AI feedback? | own clips (10-15) | hallucination rate (auto + human audit), specificity rubric, stability over 3 runs |
| RQ0 | Are the Tier-1 measurements valid? | own clips (gaze labels), RecruitView dimensions | accuracy / κ, ρ per dimension |

---

## 🧭 Pipeline

```
video -> Tier 1 measurement (MediaPipe, WhisperX, Parselmouth; aspect-ratio fixed)
           |
           |-- Part 1  score prediction (trained models)      -> RQ1, RQ2
           |-- Part 2  AI feedback (Gemini B / C / D)         -> RQ3
           '-- Part 0  measurement validity                   -> RQ0
```

### Part 1 — score prediction (core of the "train a model" work)
| ID | Method | Status |
|---|---|---|
| M0 | answer length only (baseline) | ✅ `train_score_model.py` group `length` |
| M1 | hand-set rules (old Approach A) | ✅ `rule` |
| M2 | Tier-1 features + ridge (Approach 1b) | ✅ `audio_text` / `visual` / `all`, `approaches/approach_1b_learned/` |
| M3 | pretrained embeddings + small regressor (text first, then audio) | ⏳ |
| M4 | fusion M2 + M3 | ⏳ optional |
| M5 | local LLM (Ollama, 3-4B, 4 GB VRAM) scores the transcript zero-shot | ⏳ optional — data never leaves the machine |

Protocol: train on avi_train*, choose on avi_dev, write the decision in `report/model_decision.md`,
then `--final` ONCE on avi_eval. Cross-dataset: train on AVI, test on RecruitView interview_score (evaluation only).

### Part 2 — AI feedback (own clips only)
| ID | Method | Status |
|---|---|---|
| B | Gemini, video only | ✅ `approach_2_mllm_zero_shot` |
| C | grounded: evidence log + transcript, cites IDs, checker + 1 revision | ✅ `approach_grounded` |
| C- | C without checker (= stored draft) | ✅ free |
| D | C + video | ✅ `--with-video` (optional) |

### Part 0 — measurement validity
- [x] Aspect-ratio bug fixed; one gaze measure (7.6 below)
- [ ] Gaze labels on own clips -> `validation/validate_gaze.py`
- [ ] Tier-1 dimensions vs RecruitView speaking / facial / confidence (evaluation only)

### Core vs optional
- **Core:** M0-M3 on AVI + ablation; B vs C on own clips; gaze validation
- **Optional:** M4, M5, D, RecruitView cross-dataset, MIT Interview, filler detection (CrisperWhisper / PodcastFillers)

### Data (model_decision.md, "Data decision v3")
One equal dataset for every method: train 100 (train1 + train2 + train3) / dev 30 / test 50, q1 + q2, full Tier 1.

### Next steps (in order)
1. [ ] Batch 1: `build_subset.py --split train --n 20 --name train3 --exclude-manifest <train1> <train2> --archive-others` -> `run_pipeline.py`
       (also re-measures all old clips after the aspect fix and transcribes the pilots)
2. [ ] Batch 2: `build_subset.py --split test --n 50 --name eval --archive-others` -> `run_pipeline.py`
3. [ ] `text_embed.py`; dev: `train_score_model.py --text-emb minilm --learning-curve`
4. [ ] Record 5-10 more own clips; label gaze (`validate_gaze.py`)
5. [ ] Part 2: B / C (/ D) on own clips; hallucination check + human audit
6. [ ] RecruitView evaluation subset (local, no Gemini) — optional
7. [ ] Final: `train_score_model.py --text-emb minilm --final` ONCE; write up

---

## 📌 Scope & data (decided 2026-10-04): RESEARCH ONLY
No product deployment; no own dataset collection (only the researcher's own clips). The system is a research prototype.
| Data | Use | Gemini? |
|---|---|---|
| AVI-Personality | main: train/dev/test for Part 1 | no (local LLM only) unless the authors approve |
| RecruitView | evaluation / comparison only; models trained on it never leave the experiments | ask the authors first |
| MIT Interview (on request, academic e-mail) | extra test set (optional) | check terms |
| PodcastFillers / CrisperWhisper | filler-word detection (research licence, optional) | - |
| Own clips (researcher = participant) | Part 2 feedback study, gaze labels | yes |

---

# 🗄️ Archive — history before the 2026-10-04 pivot

## ⏳ Phase 0: Main dataset = AVI-Personality

Why: AVI ships recruiter ratings for every participant, so system-vs-human
correlation no longer depends on recruiting our own raters. The official
subject-level split (train 452 / val 64 / test 130) already prevents
participant leakage (replaces GroupKFold by `user_no` in work plan v3).

Ground truth we use (1-5, one value per participant, rated after all 6 answers):
- **Primary:** `mean_rating_hirea` (overall interview performance / hireability)
- Secondary: Integrity, Collegiality, Social versatility, Development orientation
- **Not used:** HEXACO personality and cognitive ability — we measure observable
  behaviour, we do not predict personality

Done:
- [x] `base/dataset/avi.py` — labels, splits, clip keys, question text
- [x] `input/questions/avi_questions.json` — exact text of q1-q6
- [x] `base/dataset/build_subset.py` — stratified subset → `input/videos/` + metadata
- [x] `validation/evaluate_avi.py` — per-participant aggregation, Spearman + bootstrap CI, QWK, bias check
- [x] `base/_paths.py` — `AVI_DIR`, `QUESTIONS_DIR`, `GROUND_TRUTH_DIR`, `REPORT_DIR`
- [x] `.gitignore` — `output/` untracked (participant data must never be pushed)

To do:
- [ ] Build dev subset: `python base/dataset/build_subset.py --split val --n 30 --name dev`
- [ ] Build eval subset: `python base/dataset/build_subset.py --split test --n 50 --name eval --exclude-manifest input/ground_truth/avi_dev_subset.csv`
- [ ] Decide questions: generic only (q1, q2 — default) vs all six (6x cost; matches what recruiters saw)
- [ ] Rule: tune bands / weights on **dev (val)** only; report final numbers on **eval (test)** once
- [ ] Check that `user_agreement.pdf` is signed and allows our use (academic, no redistribution)

---

## ✅ Phase 1: Preprocessing (Done)

- [x] `generate_metadata.py` — อ่าน video → metadata.json
- [x] `extract_audio.py` — video → wav 16kHz mono
- [x] `extract_frames.py` — video → jpg 5fps @ 640px
- [x] `detect_face_hand.py` — MediaPipe Tasks API
  - Face: 478 จุด (มี iris)
  - Hand: 21 จุด/มือ
  - Pose: 33 จุด

---

## ✅ Phase 2: Features (Done)

- [x] `gaze.py` — eye contact ratio, gaze stability
- [x] `head_pose.py` — yaw/pitch/roll (robust + fallback)
- [x] `hand_gesture.py` — presence, speed, fidget
- [x] `facial_expression.py` — smile, EAR, brow
- [x] `speech_text.py` — Whisper + transcript + fillers

**แก้ปัญหา:**
- Camera matrix aspect ratio
- Neutral pose (diag(1,-1,-1))
- Euler formula (atan2)
- Ideal band threshold

---

## ✅ Phase 3: Scoring (Done)

- [x] `dimension_scores.py` — 5 dimensions → 0-100
- [x] `score_to_band()` — 5 levels (Excellent → Priority)
- [x] `fusion.py` — weighted sum + overall_band
- [x] เปลี่ยน linear → ideal band ทุก metric

---

## ✅ Phase 4: LLM Feedback (Done)

- [x] Gemini client + retry (503/429 handling)
- [x] Fallback models (3.6 → 2.5 → 2.0)
- [x] Prompt builder (ใช้ band แทนตัวเลข)
- [x] Feedback schema (EN + TH)

---

## ✅ Phase 5: Base Structure (Done)

- [x] สร้าง `base/` folder
- [x] ย้าย `preprocessing/` เข้า `base/`
- [x] `base/_paths.py` (centralized paths)
- [x] Refactor 4 ไฟล์ preprocessing ให้ใช้ `_paths`

---

## ⏳ Phase 6: Layer 1 Complete (Next — 1-2 weeks)

### 6.1 Prosody Extraction
- [x] `base/measurement/speech/prosody.py`
- [x] parselmouth: pitch, intensity, pause, HNR, jitter
- [x] Output → `output/evidence/{key}_prosody.json`

### 6.2 WhisperX Transcription
- [x] `base/measurement/speech/transcribe.py`
- [x] Upgrade Whisper → WhisperX (word-level timestamps)
- [x] Diarization (optional)
- [x] Output → `output/evidence/{key}_transcript.json`

### 6.3 Visual Events
- [x] `base/measurement/visual/gaze_events.py`
- [x] `base/measurement/visual/head_events.py`
- [x] `base/measurement/visual/hand_events.py`
- [x] `base/measurement/visual/face_events.py`

### 6.4 Build Evidence Log
- [x] `base/measurement/build_evidence.py`
- [x] รวม prosody + transcript + visual
- [x] Output → `output/evidence/{key}_evidence.json`

---

## ⏳ Phase 7: 3 Approaches (1 week)

### 7.1 Approach 1 — Rule-based (พร้อม)
- [x] รันได้ครบ pipeline
- [x] ย้ายเข้า `approaches/approach_1_rule/`

### 7.2 Approach 2 — MLLM Zero-shot
- [x] `approaches/approach_2_mllm_zero_shot/run.py`
- [x] Upload video → Gemini → parse feedback
- [x] ไม่ใช้ evidence log

### 7.3 Approach 3 — Hybrid
- [x] `approaches/approach_3_hybrid/build_evidence.py`
- [x] `approaches/approach_3_hybrid/interpretation/mllm_client.py`
- [x] `approaches/approach_3_hybrid/interpretation/prompt_builder.py`
- [x] `approaches/approach_3_hybrid/interpretation/feedback_generator.py`

### 7.4 Adapt all approaches to AVI (blocking for Phase 8)
- [x] Replace hard-coded "Tell me about yourself" in all 3 prompts with the AVI question + question-specific focus (`base/llm_common.py`)
- [x] Add required `ratings` field to `FEEDBACK_SCHEMA` (band per dimension + `overall`); A1 stores its measured bands
- [x] Same Gemini model + temperature (0.2) for all approaches via `base/llm_common.py` / `GEMINI_MODEL` in .env; model, temperature, prompt_version saved in every output; skip existing outputs unless `--force`
- [x] `extract_frames.py`: 5 fps → 30 fps default (`--fps 10` for the comparison, still to run)
- [x] `interviewer_present: false` for AVI (one-way interview) — diarization not needed
- [x] A1 speech features read the WhisperX transcript (one transcript for A1 and the evidence log)
- [ ] Re-tune pause / filler bands on dev — WhisperX finds ~1.5-2x more pauses than the old Whisper run

### 7.5 Approach 1b — Learned overall score (2026-10-04)
- [x] `approaches/approach_1b_learned/` — same Tier-1 measurements as A1; overall score from the ridge model fixed in `report/model_decision.md` (audio_text, AVI hireability); dimension bands still from A1
- [x] `train.py` saves `models/approach_1b/` (model, model card, fit participant ids — gitignored); default fit = train+dev, never the test manifest
- [x] `score.py` → `output/scores_1b/<key>.json`: percentile 0-100 + Thai band, which feature groups raised/lowered it, "outside training range" warning; no score if speech features are missing
- [x] `run_pipeline.py` step `score_1b` (skipped if no model); `evaluate_avi.py --approaches learned` drops participants the model was trained on
- [x] Dev check (fit=train): ρ = 0.27 [-0.10, 0.56], same as train_score_model audio_text → code path verified
- [ ] Fairness: on dev the 1b score separates gender more than recruiters do (SMD -0.59 vs -0.25, n=28) — check on test, consider dropping voice features if it holds
- [ ] Later: per-dimension learned models trained on RecruitView dimension scores

### 7.6 Tier-2 redesign: judge / writer / checker (2026-10-04)
New line-up: A = rules (approach_1_rule), B = MLLM video only (approach_2), C = grounded text-only, D = grounded + video (approach_grounded --with-video). approach_3_hybrid kept for the existing pilot results.
- [x] `base/grounded_log.py` — citable evidence log: S# summaries, E# visual episodes >= 1 s (flickers merged), P# pauses >= 1 s from WhisperX word gaps, T# transcript lines; fillers marked "not measured"
- [x] `approaches/approach_grounded/` — judge (A1 dimension bands + 1b overall) -> Gemini writer must cite IDs -> `verify.py` checks every point (unknown ID, wrong behaviour, time not in cited item, contradicted value, filler, quote) -> one revision round -> failing points removed
- [x] Output keeps `draft` + `verification` so the checker's effect is measured without extra API calls
- [x] `hallucination_check.py`: approaches grounded / grounded_draft / grounded_video(_draft) + paired comparisons; never overwrites a rated audit sample (writes *_new.csv)
- [ ] Run C on pilot clips (needs transcripts for pilot_01/03/04/05), then D
- [x] Tier-1 fix (2026-10-04): ASPECT-RATIO BUG. Landmarks are normalised per axis (x/width, y/height); every distance mixing x and y was distorted. Effects found: eye openness on 720x1280 phone video looked "closed" in ~80% of frames; head_events used 16:9 for every video -> "turned_up" for whole pilot clips; AVI features were computed with 16:9 although AVI is 4:3 (metadata.json lacked AVI keys). Both Approach-3 errors in the RQ1 pilot (pilot_04 gaze, pilot_02 head) trace back to this.
  - `base/measurement/visual/geometry.py`: real frame size (metadata -> cache -> probe the video file) + undistorted landmark helper; used by all visual events and Approach A features
  - ONE gaze measure (`gaze_events.frame_states`, used by features/gaze.py too): eyelid level vs the person's own open-eye level (long low lids = looking down / notes), gaze direction = head yaw + iris vs the person's usual direction; blinks ignored
  - Pilot sanity check after fix: pilot_03 (reading notes) 16% eye contact (was 100% / 9%), pilot_04 (camera) 100% (was 100% / 24%), pilot_05 76%; head on pilot_02/03/04 now "centered" (was "turned_up" all clip)
- [ ] Rerun visual steps for ALL clips: `python run_pipeline.py --from gaze`, then re-run train_score_model.py on dev (visual / all groups change; audio_text and the 1b decision are not affected) — record in model_decision.md
- [ ] Validate gaze against human labels: `validation/validate_gaze.py` (pilot clips, 2-s windows)

---

## ⏳ Phase 8: Validation (AVI ground truth)

### 8.1 Agreement with recruiters (RQ: does the score track human judgement?)
- [ ] Run A1/A2/A3 on eval subset
- [ ] `python validation/evaluate_avi.py --manifest input/ground_truth/avi_eval_subset.csv`
- [ ] Primary: Spearman (overall vs hireability) + 95% bootstrap CI
- [ ] QWK as secondary only — hireability is mostly 2.5-3.5, so rounded bands collapse to "3"
- [ ] Exploratory: each dimension vs each competency

### 8.1b Dev result (2026-10-02) + learned score (RQ2)
- [x] Approach 1 hand-set bands vs hireability on dev (n=30): ρ = -0.17 [95% CI -0.55, 0.22] -> no agreement
- [x] Raw features do carry signal (answer length ρ≈+0.3, smile ratio +0.37, pause count +0.44; exploratory, n=30)
- [x] `run_pipeline.py` — whole Tier-1 + scoring in one command, frees frame images after landmarks
- [x] `validation/train_score_model.py` — ridge on train, choose on dev, `--final` once on test; length baseline; RQ2 = 'all' vs 'audio_text'
- [x] Process train1 (40 people) + train2 (40 people)
- [x] Run train_score_model.py on dev (decision: report/model_decision.md)
- [ ] Eval subset (test) + `--final` once

### 8.2 Hallucination check (RQ1)
- [x] Extract timestamped claims from A2/A3 feedback (`validation/hallucination_check.py`)
- [x] Match against `*_evidence.json` within ±1 s → supported / contradicted / unsupported / out_of_range; paired A2 vs A3 difference with bootstrap CI
- [x] Blind audit sample export + `--score-audit` (rater kappa, auto-vs-human agreement)
- [ ] Run on real A2/A3 outputs; 2 raters fill `report/hallucination_audit_sample.csv`

### 8.3 Stability & robustness
- [ ] Rerun A2/A3 ≥3 times on the same clips → SD / ICC across runs
- [ ] Degrade video (fps / resolution) → change in A1/A3 scores

### 8.4 Fairness
- [ ] `report/avi_eval_bias.csv`: system vs human correlation with accent strength, English proficiency, gender

### 8.5 Comparison report
- [ ] Table: 3 approaches × (Spearman, QWK, hallucination rate, stability, cost/time)
- [ ] Replace LLM-as-judge in `compare_approaches.py` (unblinded, Gemini judging Gemini, cannot see video) or report it only as a secondary signal

---

## ⏳ Phase 9: Ethics & Framing

- [ ] เปลี่ยนคำ "บุคลิกภาพ" → "พฤติกรรมที่สังเกตได้"
- [ ] Positioning: self-practice tool, ไม่ใช่ hiring tool
- [ ] Limitation: bias ต่อ autism, social anxiety, culture
- [ ] แสดงผลเป็นระดับ (ไม่ใช่ 87.3)
- [ ] อ้างอิง EU AI Act
- [ ] AVI user agreement: academic use only, no video redistribution, no re-identification
- [ ] Never commit `output/` or AVI files; purge old media (`output/frames`, `output/audio` of vid_0021/vid_0042) from git history and GitHub

---

## ⏳ Phase 10: Paper Writing (2 weeks)

- [ ] Chapter 1: Introduction + Research Gap
- [ ] Chapter 2: Related Work
- [ ] Chapter 3: Methodology (3 approaches)
- [ ] Chapter 4: Experiments
- [ ] Chapter 5: Results
- [ ] Chapter 6: Discussion + Limitations
- [ ] Chapter 7: Ethics + Conclusion

---

## 📅 Timeline (Proposed)

| สัปดาห์ | งาน | Status |
|---|---|---|
| 1-2 | Preprocessing + Landmarks | ✅ |
| 3-4 | Features + Speech | ✅ |
| 5 | Scoring + LLM Feedback | ✅ |
| 6 | Base Structure Refactor | ✅ |
| 7-8 | Layer 1 (prosody + WhisperX + events) | ⏳ |
| 9 | Build A2 + A3 | ⏳ |
| 10-11 | Human ratings + Validation | ⏳ |
| 12-13 | Ethics + Paper writing | ⏳ |
| 14 | Final revision | ⏳ |

---

## 🎯 Milestones

- [x] **M1:** Preprocessing ครบ (Done — Sep 2026)
- [x] **M2:** Features ครบ (Done — Sep 2026)
- [x] **M3:** Scoring + LLM Feedback (Done — Sep 2026)
- [ ] **M4:** Evidence Log สมบูรณ์ (Phase 6)
- [ ] **M5:** 3 Approaches พร้อม (Phase 7)
- [ ] **M6:** Human rating + Correlation (Phase 8)
- [ ] **M7:** Paper draft (Phase 10)
- [ ] **M8:** Final submission

---

## ⚠️ Critical Path

**Validation ใช้เวลานานสุด** → เริ่มวางแผนเลย

- หา raters (3 คน)
- เตรียมคลิปชุดทดสอบ
- เตรียม scoring rubric
- ตกลงเกณฑ์ให้คะแนน

**อย่ารอจนถึง Phase 8** — เริ่มเตรียมตั้งแต่ Phase 6-7

---

## 📌 Notes

- **ทำเสร็จ 1 อย่าง → อัปเดตไฟล์นี้ทันที**
- **commit message:** `[Phase X.Y] <description>`
- **ก่อน push:** รัน test ว่ายังทำงานได้