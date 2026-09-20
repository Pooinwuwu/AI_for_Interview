"""features/head_pose.py — head orientation & movement (aspect-fixed v2)

แก้ปัญหา:
  1. Camera matrix ผิด aspect → pitch เพี้ยน 180°
  2. Fallback เมื่อ valid_frames = 0 → คืน key ครบทุกตัว ป้องกัน None
  3. อ่าน aspect จาก metadata อัตโนมัติ (ส่ง aspect เข้ามาได้)
"""

import numpy as np
import cv2
import statistics as st

from utils import valid_face_frames


# ---- 3D face model (physical mm) ----
MODEL_3D = np.array([
    (0.0,     0.0,    0.0),      # 1   nose tip
    (0.0,  -330.0,  -65.0),      # 152 chin
    (-225.0, 170.0, -135.0),     # 33  right eye outer
    (225.0,  170.0, -135.0),     # 263 left eye outer
    (-150.0, -150.0, -125.0),    # 61  right mouth corner
    (150.0,  -150.0, -125.0),    # 291 left mouth corner
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# ---- config ----
# ---- config ----
DEFAULT_ASPECT = 16 / 9
PITCH_RANGE = (-60, 60)   # ← ผ่อนจาก (-45, 45)
YAW_RANGE   = (-70, 70)   # ← ผ่อนจาก (-60, 60)
ROLL_RANGE  = (-45, 45)   # ← คงเดิม
MEDIAN_K    = 5


# ============================================================
# HELPERS
# ============================================================

def _build_camera_matrix(aspect: float):
    """
    Camera matrix ในหน่วย H-normalized
    - fx = fy = 1.0     (สมมติ f = H → FOV ≈ 90°)
    - cx = aspect / 2   (จุดกลางแนวนอน = W/2 ÷ H)
    - cy = 0.5          (จุดกลางแนวตั้ง = H/2 ÷ H)
    """
    return np.array([
        [1.0, 0.0, aspect / 2.0],
        [0.0, 1.0, 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _euler_from_rvec(rvec):
    """rvec → (pitch, yaw, roll) หน่วยองศา

    Neutral pose: หน้าเงยตรงกล้อง
    - Model: Y-up, Z-out-of-face
    - Camera (OpenCV): Y-down, Z-into-scene
    → ต้องหมุน 180° รอบแกน X (R_neutral = diag(1, -1, -1))
      ก่อนคำนวณ euler angle เพื่อให้ได้ "ส่วนเบี่ยงเบนจากneutral"
    """
    rmat, _ = cv2.Rodrigues(rvec)

    # ⭐ หักลบ neutral pose: negate แถว 1 และ 2 ของ rmat
    rmat = rmat * np.array([
        [1, 1, 1],
        [-1, -1, -1],
        [-1, -1, -1],
    ], dtype=np.float64)

    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        # gimbal lock
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = 0.0

    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))


def _is_valid_pose(p, y, r):
    return (PITCH_RANGE[0] < p < PITCH_RANGE[1]
            and YAW_RANGE[0] < y < YAW_RANGE[1]
            and ROLL_RANGE[0] < r < ROLL_RANGE[1])


def _median_filter(values, k):
    if len(values) < k:
        return values[:]
    half = k // 2
    return [
        st.median(values[max(0, i - half): i + half + 1])
        for i in range(len(values))
    ]


def _robust_stats(values):
    if not values:
        return 0.0, 0.0
    if len(values) < 4:
        med = st.median(values)
        s = st.pstdev(values) if len(values) > 1 else 0.0
        return med, s
    med = st.median(values)
    q1, q3 = st.quantiles(values, n=4)[0], st.quantiles(values, n=4)[2]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    clean = [v for v in values if lo <= v <= hi]
    if len(clean) < 2:
        return med, 0.0
    return med, st.pstdev(clean)


def _empty_result(rejected: int):
    """คืน key ครบทุกตัว ป้องกัน None"""
    return {
        "mean_yaw":          0.0,
        "mean_pitch":        0.0,
        "mean_roll":         0.0,
        "yaw_std":           0.0,
        "pitch_std":         0.0,
        "roll_std":          0.0,
        "yaw_range":         0.0,
        "pitch_range":       0.0,
        "valid_frames":      0,
        "rejected_outliers": rejected,
    }


# ============================================================
# MAIN
# ============================================================

def extract(landmarks_data, verbose=False, aspect=None):
    """
    Args:
        landmarks_data: dict จาก output/landmarks/{key}.json
        verbose: print สถิติการ reject
        aspect: w/h ของวิดีโอ — ถ้า None ใช้ DEFAULT_ASPECT
    """
    if aspect is None:
        aspect = DEFAULT_ASPECT

    cam_matrix = _build_camera_matrix(aspect)
    dist_coeffs = np.zeros((4, 1))

    pitches, yaws, rolls = [], [], []
    rejected = 0
    raw_pitches = []

    for idx, face in valid_face_frames(landmarks_data, min_points=468):
        # ⭐ คูณ x ด้วย aspect ให้ unit เท่ากับ y
        image_pts = np.array(
            [[face[i]["x"] * aspect, face[i]["y"]] for i in LANDMARK_IDS],
            dtype=np.float64,
        )
        ok, rvec, _ = cv2.solvePnP(
            MODEL_3D, image_pts, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            rejected += 1
            continue

        p, y, r = _euler_from_rvec(rvec)
        raw_pitches.append(p)

        if not _is_valid_pose(p, y, r):
            rejected += 1
            continue

        pitches.append(p); yaws.append(y); rolls.append(r)

    total = len(pitches) + rejected

    if verbose:
        print(f"  [head_pose] aspect={aspect:.4f}  "
            f"kept {len(pitches)}/{total}  (rejected {rejected})")
        if raw_pitches:
            print(f"  [head_pose] raw pitch — "
                f"min={min(raw_pitches):.2f}  max={max(raw_pitches):.2f}  "
                f"mean={st.mean(raw_pitches):.2f}  "
                f"median={st.median(raw_pitches):.2f}")
        if pitches:
            print(f"  [head_pose] kept stats — "
                f"pitch median={st.median(pitches):.2f}  "
                f"yaw median={st.median(yaws):.2f}  "
                f"roll median={st.median(rolls):.2f}")

    if not pitches:
        return _empty_result(rejected)

    pitches_s = _median_filter(pitches, MEDIAN_K)
    yaws_s    = _median_filter(yaws, MEDIAN_K)
    rolls_s   = _median_filter(rolls, MEDIAN_K)

    mean_pitch, std_pitch = _robust_stats(pitches_s)
    mean_yaw,   std_yaw   = _robust_stats(yaws_s)
    mean_roll,  std_roll  = _robust_stats(rolls_s)

    return {
        "mean_yaw":          round(mean_yaw, 2),
        "mean_pitch":        round(mean_pitch, 2),
        "mean_roll":         round(mean_roll, 2),
        "yaw_std":           round(std_yaw, 2),
        "pitch_std":         round(std_pitch, 2),
        "roll_std":          round(std_roll, 2),
        "yaw_range":         round(max(yaws_s) - min(yaws_s), 2),
        "pitch_range":       round(max(pitches_s) - min(pitches_s), 2),
        "valid_frames":      len(pitches),
        "rejected_outliers": rejected,
    }