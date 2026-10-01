# Roadmap — Phase by Phase

> แผนงานทั้งหมด + progress tracking

**Overall Progress:** ~40% complete  
**Last Updated:** 2026-10-01  
**Main dataset:** AVI-Personality (646 participants, 3,876 clips, recruiter-rated ground truth) — see Phase 0

---

## 📊 Summary

| Phase | Status | Progress | ETA |
|---|---|---|---|
| 0. Switch to AVI-Personality | ⏳ | 60% | 2-3 days |
| 1. Preprocessing | ✅ Done | 100% | — |
| 2. Features | ✅ Done | 100% | — |
| 3. Scoring | ✅ Done | 100% | — |
| 4. LLM Feedback | ✅ Done | 100% | — |
| 5. Base Structure | ✅ Done | 100% | — |
| 6. Layer 1 Complete | ⏳ | 0% | 1-2 weeks |
| 7. Build A2 + A3 | ⏳ | 30% | 1 week |
| 8. Validation | ⏳ | 0% | 2-3 weeks |
| 9. Ethics + Framing | ⏳ | 0% | — |
| 10. Paper Writing | ⏳ | 0% | 2 weeks |

---

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
- [ ] Replace hard-coded "Tell me about yourself" in all 3 prompts with `question_for_key(key)` (AVI q1-q6)
- [ ] Add required `ratings` field to `FEEDBACK_SCHEMA` (band per dimension + `overall`) so A2/A3 can be scored against ground truth
- [ ] Same Gemini model + temperature for all approaches; no silent fallback (drop and rerun a clip instead); log model per output
- [ ] `extract_frames.py`: 5 fps → 30 fps default, compare 10 fps on a subset (work plan v3 §2.2)
- [ ] `interviewer_present: false` for AVI (one-way interview) — diarization not needed

---

## ⏳ Phase 8: Validation (AVI ground truth)

### 8.1 Agreement with recruiters (RQ: does the score track human judgement?)
- [ ] Run A1/A2/A3 on eval subset
- [ ] `python validation/evaluate_avi.py --manifest input/ground_truth/avi_eval_subset.csv`
- [ ] Primary: Spearman (overall vs hireability) + 95% bootstrap CI
- [ ] QWK as secondary only — hireability is mostly 2.5-3.5, so rounded bands collapse to "3"
- [ ] Exploratory: each dimension vs each competency

### 8.2 Hallucination check (RQ1)
- [ ] Extract timestamped claims from A2/A3 feedback
- [ ] Match against `*_evidence.json` within a fixed tolerance window
- [ ] Human check on a sample (2 people, blind to approach) — this is where our own raters are still needed

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