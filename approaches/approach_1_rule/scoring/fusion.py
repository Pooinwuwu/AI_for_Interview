"""
scoring/fusion.py — รวมคะแนน 5 ด้าน เป็น overall score
"""

from dimension_scores import score_to_band


# ---- น้ำหนัก (ปรับได้ตามงานวิจัย) ----
DEFAULT_WEIGHTS = {
    "eye_contact":        0.20,
    "head_pose":          0.15,
    "hand_gesture":       0.10,
    "facial_expression":  0.15,
    "answer_quality":     0.40,
}


def weighted_sum(dimension_scores: dict, weights=None) -> dict:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total = 0.0
    weight_sum = 0.0
    breakdown = {}

    for key, w in weights.items():
        dim = dimension_scores.get(key, {})
        s = dim.get("score", 0.0)
        band = dim.get("band", {})

        total += s * w
        weight_sum += w

        breakdown[key] = {
            "score": round(s, 1),
            "band_label_th": band.get("label_th", "-"),
            "band_label_en": band.get("label_en", "-"),
            "weight": w,
            "contribution": round(s * w, 2),
        }

    overall = total / weight_sum if weight_sum > 0 else 0.0
    overall = round(overall, 1)

    return {
        "overall_score": overall,
        "overall_band":  score_to_band(overall),
        "breakdown":     breakdown,
        "weights_used":  weights,
    }