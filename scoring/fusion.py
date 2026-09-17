"""
scoring/fusion.py — รวมคะแนน 5 ด้าน เป็น overall score
"""

# ---- น้ำหนัก (ปรับได้ตามงานวิจัย) ----
# ใช้ตามสัดส่วนความสำคัญที่ literature แนะนำ
DEFAULT_WEIGHTS = {
    "eye_contact":        0.20,
    "head_pose":          0.15,
    "hand_gesture":       0.10,
    "facial_expression":  0.15,
    "answer_quality":     0.40,   # เนื้อหาสำคัญที่สุด
}


def weighted_sum(dimension_scores: dict, weights=None) -> dict:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total = 0.0
    weight_sum = 0.0
    breakdown = {}

    for key, w in weights.items():
        s = dimension_scores.get(key, {}).get("score", 0.0)
        total += s * w
        weight_sum += w
        breakdown[key] = {
            "score": round(s, 1),
            "weight": w,
            "contribution": round(s * w, 2),
        }

    if weight_sum > 0:
        overall = total / weight_sum
    else:
        overall = 0.0

    return {
        "overall_score": round(overall, 1),
        "breakdown": breakdown,
        "weights_used": weights,
    }