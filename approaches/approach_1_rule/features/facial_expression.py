"""features/facial_expression.py — smile, brow, eye openness"""

from utils import dist_2d, lm, mean, std, valid_face_frames

# MediaPipe Face Mesh indices
MOUTH_L = 61
MOUTH_R = 291
MOUTH_TOP = 13
MOUTH_BOT = 14

BROW_L_IN = 336
BROW_L_OUT = 300
BROW_R_IN = 107
BROW_R_OUT = 70

EYE_L_TOP = 386
EYE_L_BOT = 374
EYE_L_OUT = 362
EYE_L_IN  = 263

EYE_R_TOP = 159
EYE_R_BOT = 145
EYE_R_OUT = 33
EYE_R_IN  = 133

FACE_W_L = 33
FACE_W_R = 263

# threshold สำหรับนับ smile
SMILE_MOUTH_W_RATIO = 0.45    # mouth_w / face_w
SMILE_MAR_MIN       = 0.05


def extract(landmarks_data):
    mars = []       # mouth aspect ratio
    mouth_w_ratio = []   # mouth_w / face_w
    brow_h = []
    ears = []       # eye aspect ratio
    smile_count = 0
    valid = 0

    for _, face in valid_face_frames(landmarks_data, min_points=468):
        mw = dist_2d(lm(face, MOUTH_L), lm(face, MOUTH_R))
        mh = dist_2d(lm(face, MOUTH_TOP), lm(face, MOUTH_BOT))
        fw = dist_2d(lm(face, FACE_W_L), lm(face, FACE_W_R))
        if mw < 1e-6 or fw < 1e-6:
            continue

        mar = mh / mw
        mwr = mw / fw
        mars.append(mar)
        mouth_w_ratio.append(mwr)
        valid += 1

        if mwr > SMILE_MOUTH_W_RATIO and mar > SMILE_MAR_MIN:
            smile_count += 1

        # brow height (relative to eye)
        bl = dist_2d(lm(face, BROW_L_IN), lm(face, EYE_L_TOP))
        br = dist_2d(lm(face, BROW_R_IN), lm(face, EYE_R_TOP))
        brow_h.append(mean([bl, br]))

        # eye aspect ratio
        lh = dist_2d(lm(face, EYE_L_TOP), lm(face, EYE_L_BOT))
        lw = dist_2d(lm(face, EYE_L_OUT), lm(face, EYE_L_IN))
        rh = dist_2d(lm(face, EYE_R_TOP), lm(face, EYE_R_BOT))
        rw = dist_2d(lm(face, EYE_R_OUT), lm(face, EYE_R_IN))
        if lw > 1e-6 and rw > 1e-6:
            ears.append(mean([lh / lw, rh / rw]))

    if valid == 0:
        return {"valid_frames": 0}

    return {
        "mouth_aspect_ratio_mean": round(mean(mars), 4),
        "mouth_aspect_ratio_std":  round(std(mars), 4),
        "mouth_width_ratio_mean":  round(mean(mouth_w_ratio), 4),
        "mouth_width_ratio_std":   round(std(mouth_w_ratio), 4),
        "smile_ratio":             round(smile_count / valid, 4),
        "brow_height_mean":        round(mean(brow_h), 4),
        "brow_height_std":         round(std(brow_h), 4),
        "eye_aspect_ratio_mean":   round(mean(ears), 4) if ears else 0.0,
        "expressiveness":          round(std(mars) + std(mouth_w_ratio), 4),
        "valid_frames":            valid,
    }