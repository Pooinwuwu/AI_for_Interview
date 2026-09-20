"""
diagnose_head_pose.py  (v2 — compatible with head_pose.py aspect-fixed)

Usage:
    python features/diagnose_head_pose.py output/landmarks/vid_0021.json
"""

import sys
import json
import statistics as st
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_pose
from utils import get_video_aspect


def load_landmarks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_all_poses(landmarks_data, aspect):
    """คำนวณ pitch/yaw/roll ทุกเฟรม (ไม่มี filter)"""
    cam_matrix = head_pose._build_camera_matrix(aspect)
    dist_coeffs = np.zeros((4, 1))
    rows = []

    for idx, face in head_pose.valid_face_frames(landmarks_data, min_points=468):
        image_pts = np.array(
            [[face[i]["x"] * aspect, face[i]["y"]] for i in head_pose.LANDMARK_IDS],
            dtype=np.float64,
        )
        ok, rvec, _ = cv2.solvePnP(
            head_pose.MODEL_3D, image_pts,
            cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        p, y, r = head_pose._euler_from_rvec(rvec)

        ts = landmarks_data["frames"][idx].get("timestamp")
        rows.append({
            "frame_idx": idx,
            "frame_file": landmarks_data["frames"][idx]["frame_file"],
            "timestamp": ts,
            "pitch": p, "yaw": y, "roll": r,
        })
    return rows


def probe_one_frame(landmarks_data, aspect):
    """แสดง landmark 6 จุดแรก + rotation matrix ของเฟรมแรก"""
    print(f"\n[PROBE] aspect = {aspect:.4f}")
    print(f"        camera: fx=fy=1.0, cx={aspect/2:.4f}, cy=0.5")

    for i, f in enumerate(landmarks_data["frames"]):
        face = f.get("face")
        if face and len(face) >= 468:
            print(f"\n  frame#{i}: {f['frame_file']}")
            print(f"  {'idx':>5s}  {'x_mp':>8s}  {'y_mp':>8s}  {'x*asp':>8s}")

            for idx_lm in head_pose.LANDMARK_IDS:
                x_mp = face[idx_lm]["x"]
                y_mp = face[idx_lm]["y"]
                print(f"  {idx_lm:>5d}  {x_mp:>8.4f}  {y_mp:>8.4f}  "
                      f"{x_mp * aspect:>8.4f}")

            # ลอง solvePnP เฟรมนี้เลย
            image_pts = np.array(
                [[face[k]["x"] * aspect, face[k]["y"]]
                 for k in head_pose.LANDMARK_IDS],
                dtype=np.float64,
            )
            cam_matrix = head_pose._build_camera_matrix(aspect)
            ok, rvec, _ = cv2.solvePnP(
                head_pose.MODEL_3D, image_pts,
                cam_matrix, np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                rmat, _ = cv2.Rodrigues(rvec)
                p, y, r = head_pose._euler_from_rvec(rvec)
                print(f"\n  rotation matrix:")
                for row in rmat:
                    print(f"    [{row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f}]")
                print(f"  det = {np.linalg.det(rmat):+.4f}")
                print(f"  euler = pitch={p:+.2f}  yaw={y:+.2f}  roll={r:+.2f}")
            return
    print("  (ไม่พบเฟรมที่มี face)")


def describe(name, values):
    if not values:
        print(f"  {name:6s}: (ว่าง)")
        return
    print(f"  {name:6s}: n={len(values):4d}  "
          f"min={min(values):8.2f}  max={max(values):8.2f}  "
          f"mean={st.mean(values):8.2f}  median={st.median(values):8.2f}  "
          f"std={st.pstdev(values):8.2f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_head_pose.py <landmarks.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[ERROR] ไม่พบไฟล์: {path}")
        sys.exit(1)

    key = path.stem
    aspect = get_video_aspect(key)

    print("=" * 70)
    print(f"DIAGNOSE: {path.name}   (aspect={aspect:.4f})")
    print("=" * 70)

    data = load_landmarks(path)

    print("\n[0] PROBE เฟรมแรก")
    probe_one_frame(data, aspect)

    rows = scan_all_poses(data, aspect)
    if not rows:
        print("\nไม่พบข้อมูล face เลย")
        sys.exit(1)

    pitches = [r["pitch"] for r in rows]
    yaws    = [r["yaw"]   for r in rows]
    rolls   = [r["roll"]  for r in rows]

    print("\n[1] RAW statistics (ไม่มี filter)")
    describe("pitch", pitches)
    describe("yaw",   yaws)
    describe("roll",  rolls)

    # ---------- 2. Outliers ----------
    print("\n[2] Outlier count ตาม sanity range ของ head_pose")
    p_out = [r for r in rows
             if not (head_pose.PITCH_RANGE[0] < r["pitch"] < head_pose.PITCH_RANGE[1])]
    y_out = [r for r in rows
             if not (head_pose.YAW_RANGE[0] < r["yaw"] < head_pose.YAW_RANGE[1])]
    r_out = [r for r in rows
             if not (head_pose.ROLL_RANGE[0] < r["roll"] < head_pose.ROLL_RANGE[1])]
    n = len(rows)
    print(f"  pitch นอกช่วง {head_pose.PITCH_RANGE}: {len(p_out)}/{n} ({100*len(p_out)/n:.1f}%)")
    print(f"  yaw   นอกช่วง {head_pose.YAW_RANGE}:   {len(y_out)}/{n} ({100*len(y_out)/n:.1f}%)")
    print(f"  roll  นอกช่วง {head_pose.ROLL_RANGE}:  {len(r_out)}/{n} ({100*len(r_out)/n:.1f}%)")

    rejected_all = len([r for r in rows
                        if r in p_out or r in y_out or r in r_out])
    print(f"  reject ทั้งหมด: {rejected_all}/{n} ({100*rejected_all/n:.1f}%)")

    # ---------- 3. Top 5 ----------
    print("\n[3] Top 5 |pitch| สูงสุด")
    top = sorted(rows, key=lambda x: abs(x["pitch"]), reverse=True)[:5]
    for r in top:
        ts_str = f"{r['timestamp']:.2f}s" if r["timestamp"] is not None else "?"
        print(f"  frame#{r['frame_idx']:5d}  t={ts_str:>8s}  "
              f"pitch={r['pitch']:8.2f}  yaw={r['yaw']:8.2f}  roll={r['roll']:8.2f}")

    # ---------- 4. Summary ----------
    print("\n" + "=" * 70)
    print("สรุป")
    print("=" * 70)
    raw_pitch_std = st.pstdev(pitches)

    if abs(st.median(pitches)) > 30:
        print(f"⚠️  median pitch = {st.median(pitches):.2f}° → มี systematic bias")
        print("    → ตรวจ rotation matrix ใน [0] PROBE")
        print("    → ถ้า rmat[0,0] ≈ -1 (ไม่ใช่ +1) → model convention กลับด้าน")
    elif raw_pitch_std > 30:
        print(f"⚠️  pitch_std = {raw_pitch_std:.2f} → มี outlier ปนอยู่")
    else:
        print("✅ pitch อยู่ในช่วงปกติ")

    if rejected_all == n:
        print(f"⚠️  reject 100% → sanity range เข้มเกินไป")
        print(f"    → ดูค่าจริงของ yaw, roll ใน [1] แล้วปรับ PITCH_RANGE/YAW_RANGE/ROLL_RANGE")


if __name__ == "__main__":
    main()