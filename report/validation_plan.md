# Validation Plan & Strategy (Phase 8)

This document outlines the concrete steps to execute Phase 8: Validation. The goal is to rigorously compare our 3 AI approaches against human baseline scores.

## 1. Ground Truth: AVI-Personality (updated 2026-10-01)
We no longer need to collect our own ratings for score validation. AVI-Personality
provides recruiter ratings (1-5 BARS) for every participant:

*   **Primary target:** `mean_rating_hirea` — overall interview performance / hireability.
*   **Secondary:** Integrity, Collegiality, Social versatility, Development orientation.
*   **Not used:** HEXACO personality, cognitive ability (we measure observable behaviour only).
*   **Unit of analysis:** the participant. Recruiters rated after watching all six answers,
    so clip scores are averaged per participant before comparison.
*   **Splits:** official subject-level split. Tune on a dev subset from `val`;
    report once on an eval subset from `test` (stratified by hireability, default 50 people).

Scripts: `base/dataset/build_subset.py` (select + copy clips), `validation/evaluate_avi.py` (metrics).

### Where our own raters are still needed
*   **Hallucination check (RQ1):** do the timestamped claims in feedback match the video?
    2 raters, blind to which approach wrote the feedback.
*   **Feedback quality:** usefulness/specificity of coaching text (AVI has no labels for this).

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
1. [ ] Build dev (val) and eval (test) subsets with `build_subset.py`.
2. [ ] Add a `ratings` field to the feedback schema so Approach 2 & 3 can be scored.
3. [ ] Run `evaluate_avi.py`; report Spearman + 95% CI as the primary result.
4. [ ] QWK is secondary: hireability clusters around 3, so rounded bands carry little information.
5. [ ] Write the hallucination-check script and rubric for the 2 blind raters.
