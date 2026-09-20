"""
scoring/dimension_scores.py

แปลง features -> คะแนน 0-100 สำหรับ 5 ด้าน:
  1. eye_contact       (การสบตา)
  2. head_pose         (การขยับศีรษะ)
  3. hand_gesture      (การขยับมือ)
  4. facial_expression (สีหน้า)
  5. answer_quality    (การตอบคำถาม)

หลักการ:
  - ใช้ ideal band: ให้คะแนนเต็มในช่วงที่เหมาะสม ลดลงเมื่อออกนอกช่วง
  - clamp [0, 100] เสมอ
  - เพิ่ม band (Excellent/Good/Fair/Needs Work/Priority) สำหรับ user-facing
"""

import math


# ============================================================
# HELPERS
# ============================================================

def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def piecewise(value, points):
    """
    Piecewise linear interpolation
    points: list of (x, y) เรียงตาม x
    """
    if not points:
        return 0.0
    if value <= points[0][0]:
        return float(points[0][1])
    if value >= points[-1][0]:
        return float(points[-1][1])
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return 0.0


def score_to_band(score: float) -> dict:
    """
    แปลงคะแนน 0-100 → ระดับคุณภาพ 5 ระดับ
    
    Bands:
      85-100  Excellent    ดีมาก
      70-84   Good         ดี
      55-69   Fair         ปานกลาง
      40-54   Needs Work   ควรปรับ
      0-39    Priority     ควรปรับมาก
    """
    if score >= 85:
        return {"label_en": "Excellent",  "label_th": "ดีมาก",
                "color": "#4CAF50", "priority": "none"}
    if score >= 70:
        return {"label_en": "Good",       "label_th": "ดี",
                "color": "#8BC34A", "priority": "low"}
    if score >= 55:
        return {"label_en": "Fair",       "label_th": "ปานกลาง",
                "color": "#FFC107", "priority": "medium"}
    if score >= 40:
        return {"label_en": "Needs Work", "label_th": "ควรปรับ",
                "color": "#FF9800", "priority": "high"}
    return {"label_en": "Priority",   "label_th": "ควรปรับมาก",
            "color": "#F44336", "priority": "critical"}


def _wrap_dimension(score: float, components: dict, raw: dict) -> dict:
    """ห่อผลลัพธ์ของ dimension ให้มี score + band + components + raw"""
    s = round(clamp(score), 1)
    return {
        "score": s,
        "band":  score_to_band(s),
        "components": {k: round(v, 1) for k, v in components.items()},
        "raw": raw,
    }


# ============================================================
# 1. EYE CONTACT
# ============================================================

def score_eye_contact(gaze: dict) -> dict:
    """
    อ้างอิง:
      - Eye contact 55-75% = ideal (Gada et al., 2021)
      - > 90% = จ้องเข็มง อาจทำให้อีกฝ่ายอึดอัด
      - gaze stability: ideal 0.01-0.05 (นิ่งแต่ไม่แข็ง)
    """
    ratio  = gaze.get("eye_contact_ratio", 0.0)
    stab_x = gaze.get("gaze_stability_x", 0.0)
    stab_y = gaze.get("gaze_stability_y", 0.0)

    # ideal band 55-75%
    ratio_score = piecewise(ratio, [
        (0.00,   0),
        (0.30,  40),
        (0.55, 100),      # ideal เริ่ม
        (0.75, 100),      # ideal จบ
        (0.85,  90),      # เริ่มจ้อง
        (1.00,  70),      # จ้องเข็มง
    ])

    # gaze stability: ideal 0.01-0.05
    stab_mean = (stab_x + stab_y) / 2
    stab_score = piecewise(stab_mean, [
        (0.00,  70),      # แข็งเกิน
        (0.01, 100),      # ideal เริ่ม
        (0.05, 100),      # ideal จบ
        (0.10,  75),
        (0.15,  55),
        (0.25,  25),
        (0.40,   0),
    ])

    final = 0.7 * ratio_score + 0.3 * stab_score
    return _wrap_dimension(
        final,
        components={
            "eye_contact_ratio": ratio_score,
            "gaze_stability":    stab_score,
        },
        raw={
            "eye_contact_ratio": ratio,
            "gaze_stability_x":  stab_x,
            "gaze_stability_y":  stab_y,
        },
    )


# ============================================================
# 2. HEAD POSE
# ============================================================

def score_head_pose(head: dict) -> dict:
    """
    Movement std: ideal 3-8°
    Mean deviation: ideal ±10° (pitch), ±15° (yaw)
    """
    pitch_std  = head.get("pitch_std", 0.0)
    yaw_std    = head.get("yaw_std", 0.0)
    mean_pitch = abs(head.get("mean_pitch", 0.0))
    mean_yaw   = abs(head.get("mean_yaw", 0.0))

    def movement_score(std_val):
        return piecewise(std_val, [
            (0.0,  40),
            (1.5,  75),
            (3.0, 100),      # ideal
            (8.0, 100),      # ideal
            (12.0, 70),
            (20.0, 30),
            (35.0,  0),
        ])

    pitch_score = movement_score(pitch_std)
    yaw_score   = movement_score(yaw_std)

    # mean pitch: ideal ±10°
    mean_pitch_score = piecewise(mean_pitch, [
        (0.0, 100),
        (10.0, 100),     # ideal จบ
        (15.0,  85),
        (25.0,  55),
        (40.0,  20),
        (55.0,   0),
    ])

    # mean yaw: ideal ±15°
    mean_yaw_score = piecewise(mean_yaw, [
        (0.0, 100),
        (15.0, 100),     # ideal จบ
        (25.0,  80),
        (40.0,  50),
        (55.0,  15),
        (70.0,   0),
    ])

    final = (
        0.30 * pitch_score +
        0.25 * yaw_score +
        0.20 * mean_pitch_score +
        0.15 * mean_yaw_score +
        0.10 * 100
    )
    return _wrap_dimension(
        final,
        components={
            "pitch_movement":  pitch_score,
            "yaw_movement":    yaw_score,
            "pitch_deviation": mean_pitch_score,
            "yaw_deviation":   mean_yaw_score,
        },
        raw={
            "pitch_std":  pitch_std,
            "yaw_std":    yaw_std,
            "mean_pitch": head.get("mean_pitch", 0.0),
            "mean_yaw":   head.get("mean_yaw", 0.0),
        },
    )


# ============================================================
# 3. HAND GESTURE
# ============================================================

def score_hand_gesture(hand: dict) -> dict:
    """
    Presence: ideal 30-60%
    Speed: ideal 0.05-0.15
    Fidget: ideal 0-0.05 (ขยับเล็กน้อยเป็นธรรมชาติ)
    """
    presence = hand.get("hand_presence_ratio", 0.0)
    speed    = hand.get("movement_speed_mean", 0.0)
    fidget_x = hand.get("fidget_x", 0.0)
    fidget_y = hand.get("fidget_y", 0.0)

    presence_score = piecewise(presence, [
        (0.00,  30),
        (0.15,  60),
        (0.30, 100),
        (0.60, 100),
        (0.80,  75),
        (1.00,  50),
    ])

    speed_score = piecewise(speed, [
        (0.00, 50),
        (0.02, 75),
        (0.05, 100),
        (0.15, 100),
        (0.30, 60),
        (0.60, 20),
        (1.00,  0),
    ])

    # fidget: ideal 0-0.05
    fidget_mean = (fidget_x + fidget_y) / 2
    fidget_score = piecewise(fidget_mean, [
        (0.00, 100),     # ideal
        (0.05, 100),     # ideal จบ
        (0.10,  85),
        (0.20,  60),
        (0.35,  30),
        (0.50,   0),
    ])

    final = 0.45 * presence_score + 0.30 * speed_score + 0.25 * fidget_score
    return _wrap_dimension(
        final,
        components={
            "presence":       presence_score,
            "movement_speed": speed_score,
            "fidget":         fidget_score,
        },
        raw={
            "hand_presence_ratio": presence,
            "movement_speed_mean": speed,
            "fidget_x":            fidget_x,
            "fidget_y":            fidget_y,
        },
    )


# ============================================================
# 4. FACIAL EXPRESSION
# ============================================================

def score_facial_expression(face: dict) -> dict:
    """
    Smile: ideal 20-45%
    Expressiveness: ideal 0.06-0.12
    EAR: ideal 0.32-0.40
    Brow movement: ideal 0.010-0.020
    """
    smile      = face.get("smile_ratio", 0.0)
    expr       = face.get("expressiveness", 0.0)
    ear        = face.get("eye_aspect_ratio_mean", 0.0)
    brow_std   = face.get("brow_height_std", 0.0)

    smile_score = piecewise(smile, [
        (0.00,  40),
        (0.10,  75),
        (0.20, 100),     # ideal
        (0.45, 100),     # ideal
        (0.65,  70),
        (0.85,  30),
    ])

    expr_score = piecewise(expr, [
        (0.00,  40),
        (0.03,  75),
        (0.06, 100),
        (0.12, 100),
        (0.20,  60),
        (0.35,  20),
    ])

    ear_score = piecewise(ear, [
        (0.15,  30),
        (0.25,  70),
        (0.32, 100),     # ideal
        (0.40, 100),     # ideal
        (0.50,  60),
    ])

    brow_score = piecewise(brow_std, [
        (0.00,  70),
        (0.005, 90),
        (0.015, 100),    # ideal
        (0.030,  70),
        (0.050,  30),
        (0.080,   0),
    ])

    final = (
        0.35 * smile_score +
        0.25 * expr_score +
        0.25 * ear_score +
        0.15 * brow_score
    )
    return _wrap_dimension(
        final,
        components={
            "smile":          smile_score,
            "expressiveness": expr_score,
            "eye_openness":   ear_score,
            "brow_movement":  brow_score,
        },
        raw={
            "smile_ratio":           smile,
            "expressiveness":        expr,
            "eye_aspect_ratio_mean": ear,
            "brow_height_std":       brow_std,
        },
    )


# ============================================================
# 5. ANSWER QUALITY
# ============================================================

def score_answer_quality(speech: dict) -> dict:
    """
    speech rate: ideal 130-170 wpm
    filler ratio: ideal 0-0.02
    pause freq: ideal 6-10/min
    long pause: ideal 0-0.05
    """
    feats = speech.get("features", {}) if speech else {}
    wpm          = feats.get("speech_rate_wpm", 0.0)
    filler_ratio = feats.get("filler_ratio", 0.0)
    pause_count  = feats.get("pause_count", 0)
    duration     = feats.get("duration_sec", 1.0)
    long_pauses  = feats.get("long_pause_count", 0)

    wpm_score = piecewise(wpm, [
        (0,     0),
        (60,   20),
        (100,  60),
        (130, 100),
        (170, 100),
        (210,  75),
        (260,  40),
        (350,   0),
    ])

    # filler: ideal 0-0.02
    filler_score = piecewise(filler_ratio, [
        (0.00, 100),
        (0.02, 100),
        (0.05,  85),
        (0.10,  60),
        (0.15,  30),
        (0.25,   0),
    ])

    pause_per_min = pause_count / (duration / 60.0) if duration > 0 else 0.0
    pause_score = piecewise(pause_per_min, [
        (0,    60),
        (3,    85),
        (6,   100),
        (10,  100),
        (16,   70),
        (25,   30),
        (40,    0),
    ])

    long_ratio = long_pauses / max(pause_count, 1)
    long_score = piecewise(long_ratio, [
        (0.00, 100),
        (0.05, 100),
        (0.15,  85),
        (0.30,  60),
        (0.50,  30),
        (0.80,   0),
    ])

    final = (
        0.30 * wpm_score +
        0.30 * filler_score +
        0.25 * pause_score +
        0.15 * long_score
    )
    return _wrap_dimension(
        final,
        components={
            "speech_rate":  wpm_score,
            "filler_ratio": filler_score,
            "pause_freq":   pause_score,
            "long_pauses":  long_score,
        },
        raw={
            "speech_rate_wpm":  wpm,
            "filler_ratio":     filler_ratio,
            "pause_per_min":    round(pause_per_min, 2),
            "long_pause_ratio": round(long_ratio, 3),
        },
    )