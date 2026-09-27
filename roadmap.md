
---

## 📄 ไฟล์ที่ 3: `ROADMAP.md`

```markdown
# Roadmap — Phase by Phase

> แผนงานทั้งหมด + progress tracking

**Overall Progress:** ~40% complete  
**Last Updated:** 2026-09-26

---

## 📊 Summary

| Phase | Status | Progress | ETA |
|---|---|---|---|
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
- [ ] ย้ายเข้า `approaches/approach_1_rule/`

### 7.2 Approach 2 — MLLM Zero-shot
- [ ] `approaches/approach_2_mllm/run.py`
- [ ] Upload video → Gemini → parse feedback
- [ ] ไม่ใช้ evidence log

### 7.3 Approach 3 — Hybrid
- [ ] `approaches/approach_3_hybrid/build_evidence.py`
- [ ] `approaches/approach_3_hybrid/interpretation/mllm_client.py`
- [ ] `approaches/approach_3_hybrid/interpretation/prompt_builder.py`
- [ ] `approaches/approach_3_hybrid/interpretation/feedback_generator.py`

---

## ⏳ Phase 8: Validation (2-3 weeks — เริ่มวางแผนเลย)

### 8.1 Human Ratings
- [ ] หา 3 raters ที่มีประสบการณ์สัมภาษณ์
- [ ] ให้คะแนนคลิปชุดเดียวกัน (absolute rating)
- [ ] ถ้าไม่นิ่ง → pairwise comparison + Bradley-Terry

### 8.2 Metrics
- [ ] Inter-rater reliability (ICC)
- [ ] Spearman correlation (system vs human)
- [ ] Quadratic Weighted Kappa (QWK)
- [ ] Stability (rerun consistency)
- [ ] Robustness (ลด fps/resolution)

### 8.3 Comparison Report
- [ ] ตารางเทียบ 3 approaches
- [ ] วิเคราะห์ผล
- [ ] เขียน Discussion

---

## ⏳ Phase 9: Ethics & Framing

- [ ] เปลี่ยนคำ "บุคลิกภาพ" → "พฤติกรรมที่สังเกตได้"
- [ ] Positioning: self-practice tool, ไม่ใช่ hiring tool
- [ ] Limitation: bias ต่อ autism, social anxiety, culture
- [ ] แสดงผลเป็นระดับ (ไม่ใช่ 87.3)
- [ ] อ้างอิง EU AI Act

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