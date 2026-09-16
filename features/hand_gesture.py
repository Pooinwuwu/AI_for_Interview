"""features/hand_gesture.py — hand presence & movement"""

from utils import mean, std


def _hand_center(hand_landmarks):
    xs = [p["x"] for p in hand_landmarks]
    ys = [p["y"] for p in hand_landmarks]
    return mean(xs), mean(ys)


def extract(landmarks_data):
    frames = landmarks_data["frames"]
    total = len(frames)
    presence = 0
    centers = []      # (frame_idx, x, y)

    for i, f in enumerate(frames):
        hands = f.get("hands") or []
        if not hands:
            continue
        presence += 1
        cxs = [_hand_center(h["landmarks"]) for h in hands]
        cx = mean([c[0] for c in cxs])
        cy = mean([c[1] for c in cxs])
        centers.append((i, cx, cy))

    if not centers:
        return {
            "hand_presence_ratio": 0.0,
            "movement_speed_mean": 0.0,
            "movement_speed_std":  0.0,
            "fidget_x":            0.0,
            "fidget_y":            0.0,
            "hand_frames":         0,
            "total_frames":        total,
        }

    # ความเร็ว = displacement / dt (normalized unit / วินาที)
    speeds = []
    for k in range(1, len(centers)):
        i1, x1, y1 = centers[k - 1]
        i2, x2, y2 = centers[k]
        t1 = frames[i1].get("timestamp") or 0.0
        t2 = frames[i2].get("timestamp") or 0.0
        dt = t2 - t1
        if dt <= 0:
            continue
        d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        speeds.append(d / dt)

    xs = [c[1] for c in centers]
    ys = [c[2] for c in centers]

    return {
        "hand_presence_ratio": round(presence / total, 4),
        "movement_speed_mean": round(mean(speeds), 4) if speeds else 0.0,
        "movement_speed_std":  round(std(speeds), 4) if speeds else 0.0,
        "fidget_x":            round(std(xs), 4),
        "fidget_y":            round(std(ys), 4),
        "hand_frames":         presence,
        "total_frames":        total,
    }