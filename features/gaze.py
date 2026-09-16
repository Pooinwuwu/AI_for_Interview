"""features/gaze.py — eye contact ratio, gaze stability"""

from utils import dist_2d, lm, mean, std, valid_face_frames

# MediaPipe Face Mesh indices (478-point model with refine_landmarks=True)
EYES = {
    "right": {"outer": 33,  "inner": 133, "top": 159, "bottom": 145, "iris": 468},
    "left":  {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473},
}

# thresholds
EYE_OPEN_MIN       = 0.15        # eye aspect ratio ต่ำกว่านี้ = ปิดตา
X_CENTER_RANGE     = (0.35, 0.65)  # iris อยู่กลางแนวนอน
Y_CENTER_RANGE     = (0.30, 0.70)  # iris อยู่กลางแนวตั้ง


def _frame_iris_position(face, cfg):
    """คืน {x, y, ear} หรือ None ถ้าตาปิด/คำนวณไม่ได้"""
    top, bottom = lm(face, cfg["top"]), lm(face, cfg["bottom"])
    left, right = lm(face, cfg["outer"]), lm(face, cfg["inner"])

    eye_w = dist_2d(left, right)
    eye_h = dist_2d(top, bottom)
    if eye_w < 1e-6:
        return None

    ear = eye_h / eye_w
    if ear < EYE_OPEN_MIN:      # ตาปิด → ไม่นับ
        return None

    iris = lm(face, cfg["iris"])
    x_norm = (iris["x"] - left["x"]) / (right["x"] - left["x"] + 1e-6)
    y_norm = (iris["y"] - top["y"])  / (bottom["y"] - top["y"] + 1e-6)
    return {"x": x_norm, "y": y_norm, "ear": ear}


def extract(landmarks_data):
    x_vals, y_vals = [], []
    contact = 0
    valid = 0

    for _, face in valid_face_frames(landmarks_data, min_points=478):
        for cfg in EYES.values():
            pos = _frame_iris_position(face, cfg)
            if pos is None:
                continue
            x_vals.append(pos["x"])
            y_vals.append(pos["y"])
            valid += 1

            if (X_CENTER_RANGE[0] < pos["x"] < X_CENTER_RANGE[1]
                    and Y_CENTER_RANGE[0] < pos["y"] < Y_CENTER_RANGE[1]):
                contact += 1

    if valid == 0:
        return {"valid_frames": 0}

    return {
        "eye_contact_ratio":   round(contact / valid, 4),
        "gaze_stability_x":    round(std(x_vals), 4),   # ยิ่งน้อย ยิ่งนิ่ง
        "gaze_stability_y":    round(std(y_vals), 4),
        "mean_iris_x":         round(mean(x_vals), 4),
        "mean_iris_y":         round(mean(y_vals), 4),
        "valid_frames":        valid,
    }