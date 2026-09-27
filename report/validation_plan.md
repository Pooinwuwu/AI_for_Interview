# Validation Plan & Strategy (Phase 8)

This document outlines the concrete steps to execute Phase 8: Validation. The goal is to rigorously compare our 3 AI approaches against human baseline scores.

## 1. Human Ratings (Phase 8.1)
To prove the AI works, we need a "Ground Truth". 

### The Setup
*   **Raters:** Recruit 3 human raters who have experience interviewing candidates (e.g., HR professionals, senior managers).
*   **Dataset:** Select a benchmark set of videos (e.g., 20-30 videos) that cover a wide range of performance (good, average, and poor).
*   **Scoring Rubric:** We will ask raters to grade candidates on a **1 to 5 scale** for each of the 5 dimensions. This is much easier for humans than guessing a 0-100 score, and it maps perfectly to our AI bands:
    *   **5 (Excellent):** 85-100 - Flawless, professional, highly engaging.
    *   **4 (Good):** 70-84 - Strong, minor slip-ups but generally confident.
    *   **3 (Fair):** 55-69 - Average, passable but lacks polish.
    *   **2 (Needs Work):** 40-54 - Noticeable issues, distracting behaviors.
    *   **1 (Priority/Poor):** 0-39 - Severe issues, highly detrimental to the interview.

### Resolution Protocol
*   If raters generally agree (high reliability), we average their scores.
*   If raters strongly disagree on certain clips, we will use **Pairwise Comparison (Bradley-Terry model)**: instead of asking "what score is this?", we ask "Is Video A better than Video B?".

---

## 2. Metrics & Analytics (Phase 8.2)

We will write an evaluation script (`evaluate_metrics.py`) to calculate the following standard machine learning metrics:

1.  **Inter-Rater Reliability (ICC):** Measures how much the 3 human raters agree with each other. If humans can't agree, the AI can't be expected to agree with them either.
2.  **Spearman Rank Correlation:** Checks if the AI *ranks* the candidates in the same order as humans (e.g., does it correctly identify the best and worst candidates, even if the exact scores differ?).
3.  **Quadratic Weighted Kappa (QWK):** The industry standard for automated scoring. It heavily penalizes the AI if it is completely wrong (e.g., scoring a 1 when the human scored a 5).
4.  **Stability Testing:** We will run Approach 2 and 3 at least 3 times on the exact same video to measure variance (LLMs are non-deterministic, so we must prove they give consistent feedback).
5.  **Robustness Testing:** We will artificially degrade the video quality (lower resolution, drop FPS to 15) and see if Approach 1 and 3 break down.

---

## 3. The Comparison Report (Phase 8.3)

Ultimately, we will generate a comparison matrix to put in the final research paper:

| Metric | Appr 1 (Rule-based) | Appr 2 (Zero-shot) | Appr 3 (Hybrid) |
| :--- | :--- | :--- | :--- |
| **Accuracy (QWK)** | ? | ? | ? |
| **Correlation** | ? | ? | ? |
| **Stability (Var)** | 0.0 (Deterministic) | ? (LLM Variance) | ? (LLM Variance) |
| **Processing Time** | ~1x | ~2x (Video Upload) | ~2.5x (Both) |
| **Feedback Quality**| Generic | Highly Nuanced | Nuanced + Grounded |

### Next Immediate Action Items:
1. [ ] Finalize the human scoring rubric sheet (e.g., Google Form / Excel).
2. [ ] Identify and prepare the benchmark test set of videos.
3. [ ] Create a Python script to calculate ICC and QWK.
