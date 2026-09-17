"""
scoring/dimension_scores.py

แปลง features -> คะแนน 0-100 สำหรับ 5 ด้าน:
  1. eye_contact      (การสบตา)
  2. head_pose        (การขยับศีรษะ)
  3. hand_gesture     (การขยับมือ)
  4. facial_expression (สีหน้า)
  5. answer_quality   (การตอบคำถาม)

หลักการ:
  - ทุกสูตรอิงจาก literature + การสังเกตคลิปจริง
  - ใช้ piecewise linear: ให้คะแนนเต็มในช่วง ideal, ลดลงเมื่อออกนอกช่วง
  - clamp [0, 100] เสมอ
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
    ตัวอย่าง: [(0.0, 0), (0.5, 100), (1.0, 0)]
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


def inverse_scale(value, best, worst, best_score=100.0, worst_score=0.0):
    """ค่าเข้าใกล้ best -> คะแนนสูง; ค่าเข้าใกล้ worst -> คะแนนต่ำ"""
    if best == worst:
        return best_score
    t = (value - worst) / (best - worst)
    return clamp(worst_score + t * (best_score - worst_score))


# ============================================================
# 1. EYE CONTACT
# ============================================================

def score_eye_contact(gaze: dict) -> dict:
    """
    อ้างอิง:
      - Eye contact 60-70% ของเวลาถือว่าเหมาะสมสำหรับการสัมภาษณ์
        (Gada et al., 2021; Ho et al., 2020)
      - gaze stability ยิ่งน้อยยิ่งดี (สายตานิ่ง = มั่นใจ)
    """
    ratio = gaze.get("eye_contact_ratio", 0.0)
    stab_x = gaze.get("gaze_stability_x", 0.0)
    stab_y = gaze.get("gaze_stability_y", 0.0)

    # ---- eye_contact_ratio: 0 → 0, 0.7 → 100, 0.9+ ลดลงเล็กน้อย ----
    ratio_score = piecewise(ratio, [
        (0.00,   0),
        (0.30,  40),
        (0.55,  75),
        (0.70, 100),
        (0.85,  95),   # มองมากเกินไปอาจดูจ้อง
        (1.00,  85),
    ])

    # ---- gaze stability: ยิ่ง std น้อยยิ่งดี ----
    # std 0.02 = นิ่งมาก, std 0.15 = สั่นมาก
    stab_mean = (stab_x + stab_y) / 2
    stab_score = piecewise(stab_mean, [
        (0.00, 100),
        (0.05,  90),
        (0.10,  70),
        (0.15,  50),
        (0.25,  20),
        (0.40,   0),
    ])

    final = 0.7 * ratio_score + 0.3 * stab_score
    return {
        "score": round(clamp(final), 1),
        "components": {
            "eye_contact_ratio": round(ratio_score, 1),
            "gaze_stability":    round(stab_score, 1),
        },
        "raw": {
            "eye_contact_ratio": ratio,
            "gaze_stability_x":  stab_x,
            "gaze_stability_y":  stab_y,
        },
    }


# ============================================================
# 2. HEAD POSE
# ============================================================

def score_head_pose(head: dict) -> dict:
    """
    อ้างอิง:
      - Head movement ที่เหมาะสม: yaw ±15°, pitch ±10° (สุ่มตัวอย่างในบทสนทนา)
      - yaw/pitch std ต่ำ = นิ่ง, สูงเกินไป = ขยับมาก, ต่ำเกินไป = แข็ง
      - ideal pitch_std = 2-8°
    """
    pitch_std = head.get("pitch_std", 0.0)
    yaw_std   = head.get("yaw_std", 0.0)
    mean_pitch = abs(head.get("mean_pitch", 0.0))
    mean_yaw   = abs(head.get("mean_yaw", 0.0))

    # ---- movement std: ideal 2-10° ----
    def movement_score(std_val):
        return piecewise(std_val, [
            (0.0,  40),   # นิ่งเกินไป = แข็ง
            (1.5,  75),
            (3.0, 100),   # ideal
            (8.0, 100),
            (12.0, 70),
            (20.0, 30),
            (35.0,  0),   # ขยับมากเกินไป
        ])

    pitch_score = movement_score(pitch_std)
    yaw_score   = movement_score(yaw_std)

    # ---- mean deviation: ก้ม/เงย มากเกินไปหักคะแนน ----
    mean_pitch_score = piecewise(mean_pitch, [
        (0.0, 100),
        (5.0, 100),
        (10.0, 85),
        (20.0, 60),
        (35.0, 20),
        (50.0,  0),   # ก้ม/เงย 50° = มากเกินไป
    ])
    mean_yaw_score = piecewise(mean_yaw, [
        (0.0, 100),
        (10.0, 100),
        (20.0, 80),
        (35.0, 50),
        (50.0, 10),
        (70.0,  0),
    ])

    # ---- รวม ----
    final = (
        0.30 * pitch_score +
        0.25 * yaw_score +
        0.20 * mean_pitch_score +
        0.15 * mean_yaw_score +
        0.10 * 100   # base
    )

    return {
        "score": round(clamp(final), 1),
        "components": {
            "pitch_movement":   round(pitch_score, 1),
            "yaw_movement":     round(yaw_score, 1),
            "pitch_deviation":  round(mean_pitch_score, 1),
            "yaw_deviation":    round(mean_yaw_score, 1),
        },
        "raw": {
            "pitch_std": pitch_std,
            "yaw_std":   yaw_std,
            "mean_pitch": head.get("mean_pitch", 0.0),
            "mean_yaw":   head.get("mean_yaw", 0.0),
        },
    }


# ============================================================
# 3. HAND GESTURE
# ============================================================

def score_hand_gesture(hand: dict) -> dict:
    """
    อ้างอิง:
      - Gesture ที่เหมาะสม: ปรากฏ ~30-60% ของเวลา
      - movement speed ปานกลาง = gesturing, สูงมาก = fidgeting
      - fidget ยิ่งสูงยิ่งหักคะแนน
    """
    presence = hand.get("hand_presence_ratio", 0.0)
    speed    = hand.get("movement_speed_mean", 0.0)
    fidget_x = hand.get("fidget_x", 0.0)
    fidget_y = hand.get("fidget_y", 0.0)

    # ---- presence: 0% = แข็ง, 30-60% = ideal, 100% = มากเกินไป ----
    presence_score = piecewise(presence, [
        (0.00,  30),   # ไม่ขยับมือเลย
        (0.15,  60),
        (0.30, 100),   # ideal
        (0.60, 100),
        (0.80,  75),
        (1.00,  50),   # ขยับมือตลอดเวลา
    ])

    # ---- movement speed: ideal ~0.05-0.15 ----
    speed_score = piecewise(speed, [
        (0.00, 50),    # นิ่งเกินไป
        (0.02, 75),
        (0.05, 100),   # ideal
        (0.15, 100),
        (0.30, 60),
        (0.60, 20),
        (1.00,  0),    # ขยับเร็วมาก = fidgeting
    ])

    # ---- fidget: std ของตำแหน่งมือ ยิ่งสูงยิ่งไม่ดี ----
    fidget_mean = (fidget_x + fidget_y) / 2
    fidget_score = piecewise(fidget_mean, [
        (0.00, 100),
        (0.05, 100),
        (0.10,  85),
        (0.20,  60),
        (0.35,  30),
        (0.50,   0),
    ])

    final = 0.45 * presence_score + 0.30 * speed_score + 0.25 * fidget_score
    return {
        "score": round(clamp(final), 1),
        "components": {
            "presence":        round(presence_score, 1),
            "movement_speed":  round(speed_score, 1),
            "fidget":          round(fidget_score, 1),
        },
        "raw": {
            "hand_presence_ratio": presence,
            "movement_speed_mean": speed,
            "fidget_x":            fidget_x,
            "fidget_y":            fidget_y,
        },
    }


# ============================================================
# 4. FACIAL EXPRESSION
# ============================================================

def score_facial_expression(face: dict) -> dict:
    """
    อ้างอิง:
      - Smile ratio ~20-50% เหมาะสมในบริบทสัมภาษณ์
      - expressiveness ปานกลาง = แสดงออกพอเหมาะ
      - eye_aspect_ratio ต่ำ = ตาหรี่/เครียด
    """
    smile      = face.get("smile_ratio", 0.0)
    expr       = face.get("expressiveness", 0.0)
    ear        = face.get("eye_aspect_ratio_mean", 0.0)
    brow_std   = face.get("brow_height_std", 0.0)

    # ---- smile: ideal 0.15-0.45 ----
    smile_score = piecewise(smile, [
        (0.00,  40),   # ไม่ยิ้มเลย
        (0.10,  75),
        (0.20, 100),   # ideal
        (0.45, 100),
        (0.65,  70),
        (0.85,  30),   # ยิ้มตลอด = แปลก
    ])

    # ---- expressiveness: ideal 0.05-0.12 ----
    expr_score = piecewise(expr, [
        (0.00,  40),   # หน้าแข็ง
        (0.03,  75),
        (0.06, 100),
        (0.12, 100),
        (0.20,  60),
        (0.35,  20),
    ])

    # ---- EAR: ตาเปิด 0.30-0.40 = ปกติ ----
    ear_score = piecewise(ear, [
        (0.15,  30),   # ตาหรี่ = เครียด
        (0.25,  70),
        (0.32, 100),   # ideal
        (0.40, 100),
        (0.50,  60),   # เบิกตาเกิน
    ])

    # ---- brow movement: เล็กน้อยดี มากเกินไป = เครียด ----
    brow_score = piecewise(brow_std, [
        (0.00, 70),
        (0.005, 90),
        (0.015, 100),
        (0.030, 70),
        (0.050, 30),
        (0.080,  0),
    ])

    final = (
        0.35 * smile_score +
        0.25 * expr_score +
        0.25 * ear_score +
        0.15 * brow_score
    )
    return {
        "score": round(clamp(final), 1),
        "components": {
            "smile":         round(smile_score, 1),
            "expressiveness": round(expr_score, 1),
            "eye_openness":  round(ear_score, 1),
            "brow_movement": round(brow_score, 1),
        },
        "raw": {
            "smile_ratio":            smile,
            "expressiveness":         expr,
            "eye_aspect_ratio_mean":  ear,
            "brow_height_std":        brow_std,
        },
    }


# ============================================================
# 5. ANSWER QUALITY (จาก speech features)
# ============================================================

def score_answer_quality(speech: dict) -> dict:
    """
    ขั้นนี้ใช้ speech features เท่านั้น
    LLM จะวิเคราะห์เนื้อหาคำตอบแยกต่างหากในขั้น feedback

    อ้างอิง:
      - speech rate ภาษาไทยปกติ 120-180 wpm
      - filler ratio < 5% = ดี
      - pause ปกติ 4-10 ครั้ง/นาที
    """
    feats = speech.get("features", {}) if speech else {}
    wpm          = feats.get("speech_rate_wpm", 0.0)
    filler_ratio = feats.get("filler_ratio", 0.0)
    pause_count  = feats.get("pause_count", 0)
    duration     = feats.get("duration_sec", 1.0)
    long_pauses  = feats.get("long_pause_count", 0)

    # ---- speech rate: ideal 120-180 wpm ----
    wpm_score = piecewise(wpm, [
        (0,     0),
        (60,   20),    # ช้ามาก
        (100,  60),
        (130, 100),    # ideal
        (170, 100),
        (210,  75),    # เร็วเกิน
        (260,  40),
        (350,   0),
    ])

    # ---- filler ratio: 0 = 100, 0.05 = 80, 0.15 = 30 ----
    filler_score = piecewise(filler_ratio, [
        (0.00, 100),
        (0.02,  95),
        (0.05,  80),
        (0.10,  55),
        (0.15,  30),
        (0.25,   0),
    ])

    # ---- pause frequency (ต่อนาที) ----
    pause_per_min = pause_count / (duration / 60.0) if duration > 0 else 0.0
    pause_score = piecewise(pause_per_min, [
        (0,    60),   # ไม่หยุดเลย = ไหลไม่หยุด
        (3,    85),
        (6,   100),   # ideal
        (10,  100),
        (16,   70),
        (25,   30),
        (40,    0),
    ])

    # ---- long pause penalty ----
    long_ratio = long_pauses / max(pause_count, 1)
    long_score = piecewise(long_ratio, [
        (0.00, 100),
        (0.10,  85),
        (0.25,  60),
        (0.50,  30),
        (0.80,   0),
    ])

    final = (
        0.30 * wpm_score +
        0.30 * filler_score +
        0.25 * pause_score +
        0.15 * long_score
    )
    return {
        "score": round(clamp(final), 1),
        "components": {
            "speech_rate":   round(wpm_score, 1),
            "filler_ratio":  round(filler_score, 1),
            "pause_freq":    round(pause_score, 1),
            "long_pauses":   round(long_score, 1),
        },
        "raw": {
            "speech_rate_wpm":  wpm,
            "filler_ratio":     filler_ratio,
            "pause_per_min":    round(pause_per_min, 2),
            "long_pause_ratio": round(long_ratio, 3),
        },
    }