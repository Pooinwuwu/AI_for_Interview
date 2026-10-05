# Related work — draft (2026-10-04)

Notes for the report's related-work section. Numbers come from each paper's abstract or HTML page.
Open the full paper and check every number before citing it.

---

## 1. Evidence-grounded interview assessment (closest to Approach 3)

**E-AVI: Evidence-Grounded Multimodal Assessment for Automated Video Interviews**
Wang, Che, Xie, Du, Hua, Wang — Hong Kong PolyU, arXiv 2609.20001 (Sep 2026)
https://arxiv.org/html/2609.20001

- **Method:**
  - An MLLM (MiniCPM-o 4.5) extracts timestamped evidence items (modality, event type, time, description) from video, audio and transcript.
  - A dimension-conditioned scorer combines evidence embeddings with source embeddings.
- **Data:** RecruitView, plus a private hotel dataset (101 people).
- **Results:**
  - RecruitView: Spearman ρ 0.541, MAE 0.607. Private set: ρ 0.639.
  - Beats fine-tuned VideoLLaMA2, Qwen3-VL and MiniCPM-o.
- **Evidence audit:**
  - 870 evidence items checked by humans; 80.8% fully supported.
  - By modality: text 89.5%, video 77.4%, audio 74.8%.
- **Where it fails:** in 83.3% of the worst cases, the cause is wrong extracted evidence.
- **Evidence alone is not enough:** removing the source embeddings drops ρ from 0.541 to 0.261.
- **Ablation:** removing text hurts most; removing video or audio hurts less.

**Relevance to our project**
- Same idea as Approach 3: give the model timestamped evidence. Same failure mode as our pilot: when the evidence is wrong, the output is wrong (both A3 errors came from Tier 1).
- Differences:
  1. We measure with deterministic tools (MediaPipe, Parselmouth, WhisperX) instead of an MLLM extractor.
  2. We audit hallucination in the *coaching feedback* claim by claim against the video (RQ1).
  3. We run a controlled comparison of video-only vs video + evidence log with the same model and prompt (A2 vs A3).
- The RecruitView numbers (ρ 0.54) give a direct reference point for our RecruitView test.

---

## 2. Work on the AVI dataset (our main dataset)

AVI-Personality is the data of the ACM Multimedia **AVI Challenge**.

**Listening to the Unspoken: Exploring "365" Aspects of Multimodal Interview Performance Assessment**
arXiv 2507.22676 — 1st place, AVI Challenge 2025
https://arxiv.org/abs/2507.22676v3
- Modality-specific feature extractors, fused with a shared compression MLP, with one regression head per response, then mean pooling.
- Average MSE 0.1824 over 5 dimensions. Code: github.com/MSA-LMC/365Aspects
- Score prediction only; no feedback generation.

**Frozen Multimodal Embeddings for AI-Assisted Interview Assessment of Personality and Cognitive Ability**
Hung, Suen, Yeh, Wang — arXiv 2606.11930 (AVI Challenge 2026)
https://arxiv.org/pdf/2606.11930
- Frozen CLIP / Whisper / RoBERTa / E5 / DeBERTa encoders with small regressors, chosen because there are only about 450 training subjects.
- HEXACO MSE 0.3334 → 0.2696.
- Text (RoBERTa) was selected for every trait.

**Relevance**
- Same strategy as Approach 1b: a small learned model on top of fixed features, because the data is small.
- Their text-heavy result agrees with our RQ2 dev result (visual features add nothing over audio + text).
- Our ρ is not directly comparable to their MSE: different targets (they predict HEXACO, we predict hireability) and different metrics.

---

## 3. RecruitView (our second dataset)

**RecruitView: A Multimodal Dataset for Predicting Personality and Interview Performance for HR Applications**
Gupta, Sheth et al. — arXiv 2512.00450
https://arxiv.org/html/2512.00450v1
- 2,011 clips, 300+ participants, 76 questions.
- Clinical psychologists made about 27,000 pairwise comparisons, converted to 12 continuous scores with a nuclear-norm-regularised MNL model.
- Their CRMF model (geometric fusion): ρ 0.568. Fine-tuned MiniCPM-o: 0.510.
- Single modality: **video 0.45 > text 0.42 > audio 0.38**. All three together: +25.8%.
- Limitations stated by the authors: short clips (about 30 s), limited demographic diversity, not validated for real hiring.

**Relevance**
- Per-dimension labels let us test our dimension feedback.
- They could also train per-dimension models (the planned Approach 1c).
- Modality ranking differs from AVI (video strongest here, text strongest on AVI), so RQ2 may depend on the dataset. Test RQ2 on both.

---

## 4. Limits of video MLLMs (why Tier-1 measurement is still needed)

**VideoZeroBench** — arXiv 2604.01569 (2026)
https://arxiv.org/html/2604.01569v1
- Gemini-3-Pro answers under 17% of questions correctly.
- No model exceeds 1% when the answer *and* its spatio-temporal location must both be correct.
- Giving evidence hints raises accuracy from 9.6% to 28.4% → locating evidence is the core weakness.
- Supports Approach 3's idea: give the model the evidence instead of making it find the evidence.

**VidHalluc** — Li et al., CVPR 2025
https://openaccess.thecvf.com/content/CVPR2025/papers/Li_VidHalluc_Evaluating_Temporal_Hallucinations_in_Multimodal_Large_Language_Models_for_CVPR_2025_paper.pdf
- Temporal hallucination: most models score under 50% on telling apart actions that follow each other quickly.
- About half of those errors are seeing only one action in the whole video.

**ARGUS** — Rawal et al., ICCV 2025
https://openaccess.thecvf.com/content/ICCV2025/papers/Rawal_ARGUS_Hallucination_and_Omission_Evaluation_in_Video-LLMs_ICCV_2025_paper.pdf
- Measures both hallucination (invented content) and omission (missed content) in video-LLM descriptions.
- Could inform an omission measure for RQ1. Our check currently counts only wrong claims, not missed events.

---

## 5. AI mock-interview systems

- **SimInterview** — Nguyen et al., arXiv 2508.11873 (ICEFM 2025). https://arxiv.org/pdf/2508.11873
  - An LLM- and RAG-based multilingual mock-interview system with a talking-head interviewer.
  - Evaluated on user experience with 20 candidates.
  - Does not verify nonverbal feedback.
- **Virtual Interviewers, Real Results** — arXiv 2506.16542. https://arxiv.org/abs/2506.16542
  - AI mock technical interviews; outcomes are student readiness and confidence.
- **Kassab, Kashevnik, Shoshina (2026)**, *Big Data and Cognitive Computing* 10(4):106. https://www.mdpi.com/2504-2289/10/4/106
  - Computer-vision cues (eye contact, head motion, expressivity) plus Llama 3.2 rating the transcript on 1–5 scales.
  - Only 59 interviews (5 executives).

**Gap**
- Coaching systems seldom check whether their behavioural feedback is *true* for the video.
- Assessment systems (E-AVI, CRMF, the AVI Challenge entries) optimise score agreement, not feedback faithfulness.

---

## Positioning paragraph (draft, English for the report)

Recent work moves toward grounding multimodal models in explicit, timestamped evidence
(E-AVI) and shows that video MLLMs struggle to locate fine-grained evidence on their own
(VideoZeroBench, VidHalluc). Score-prediction work on the AVI and RecruitView datasets
reports moderate rank agreement with human raters (ρ ≈ 0.5–0.6 on RecruitView) but does not
evaluate whether the feedback given to candidates is faithful to the video. Our study
compares, under the same model and prompt, a video-only MLLM with one given a deterministic
Tier-1 evidence log, audits every timestamped coaching claim against the video, and tests
whether learned scores and visual features improve agreement with recruiters. Consistent
with E-AVI's error analysis, our pilot suggests that grounding is bounded by measurement
accuracy: both errors of the evidence-grounded approach traced back to Tier-1 mistakes.
