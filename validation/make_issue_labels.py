"""
validation/make_issue_labels.py   (RQ2: does the trained "doctor" pick the right thing to fix?)

Makes a blind labelling sheet: for each sampled AVI test clip the researcher watches the video
and writes the MAIN problem (and an optional second one) BEFORE seeing any model output.
The doctor's top issue is later compared with these labels (and with the rule-based doctor).

No scores, model outputs or recruiter ratings are written to the sheet (blind labelling).
Clips are watched locally from input/videos_archive (or input/videos); nothing leaves this computer.

Usage
  python validation/make_issue_labels.py                 # 30 clips, 10 of them also for the friend
  python validation/make_issue_labels.py --n 40 --overlap 12
Output
  report/issue_labels.csv          you fill main_issue / second_issue / confidence / note
  report/issue_labels_friend.csv   the overlap clips only, for the second rater (same columns)
  report/issue_labels_GUIDE.md     the issue list and the rules
An existing sheet is never overwritten (a *_new.csv is written instead).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base._paths import GROUND_TRUTH_DIR, INPUT_DIR, REPORT_DIR
from base.llm_common import question_context

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# The doctor reports its top issue with the same codes, so the two can be compared directly.
ISSUES = [
    ("too_short",        "คำตอบสั้นเกินไป / พูดน้อย",                 "answer too short"),
    ("no_example",       "ไม่มีตัวอย่างหรือหลักฐานประกอบ",             "no concrete example"),
    ("off_topic",        "ตอบไม่ตรงคำถาม / ไม่มีโครงสร้าง",            "off topic or no structure"),
    ("long_pauses",      "หยุดนาน ลังเล ติดขัด",                       "long pauses, hesitant"),
    ("pace",             "พูดเร็วหรือช้าเกินไป",                       "speaking too fast or slow"),
    ("eye_contact",      "ไม่ค่อยมองกล้อง",                            "little eye contact"),
    ("head_posture",     "ศีรษะ/ท่าทางไม่นิ่ง หรือก้ม/เงยมาก",          "head or posture"),
    ("facial_expression","สีหน้านิ่ง ไม่มีชีวิตชีวา",                    "flat facial expression"),
    ("hands",            "มือหรือท่าทางรบกวน",                         "distracting hands"),
    ("none",             "ไม่มีปัญหาชัดเจน",                           "no clear problem"),
]
COLUMNS = ["n", "key", "video", "question", "main_issue", "second_issue",
           "confidence (1-3)", "note"]


def find_video(key: str) -> str:
    for folder in ("videos_archive", "videos"):
        p = INPUT_DIR / folder / f"{key}.mp4"
        if p.exists():
            return str(p.relative_to(ROOT))
    return "(not found)"


def write_sheet(path: Path, rows) -> Path:
    if path.exists():
        path = path.with_name(path.stem + "_new.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def guide_text(n: int, overlap: int) -> str:
    lines = [
        "# คู่มือติดป้ายปัญหาหลัก (AVI test)",
        "",
        f"ดู {n} คลิปใน `issue_labels.csv` ทีละคลิป แล้วกรอก **ก่อน**ดูผลของโมเดลใด ๆ",
        "",
        "1. เปิดไฟล์ในคอลัมน์ `video` ดูทั้งคลิป 1 รอบ (เร่ง 1.25x ได้)",
        "2. `main_issue` = ปัญหาที่ถ้าแก้แล้วคำตอบจะดีขึ้นมากที่สุด ใส่ **รหัส 1 ตัว** จากตารางด้านล่าง",
        "3. `second_issue` = ปัญหาอันดับ 2 (เว้นว่างได้)",
        "4. `confidence` = 1 ไม่แน่ใจ · 2 ค่อนข้างแน่ใจ · 3 แน่ใจมาก",
        "5. ห้ามเปิดดูคะแนนกรรมการ หรือผลของโมเดล จนกว่าจะกรอกครบ",
        "",
        "| รหัส | ความหมาย |",
        "| --- | --- |",
    ]
    lines += [f"| `{c}` | {th} ({en}) |" for c, th, en in ISSUES]
    lines += [
        "",
        f"`issue_labels_friend.csv` มี {overlap} คลิปเดียวกัน ให้เพื่อนกรอกแยก ห้ามดูของกันและกัน "
        "ใช้วัดว่าคนสองคนเห็นตรงกันแค่ไหน (Cohen's kappa)",
        "",
        "ข้อมูล AVI: ดูในเครื่องนี้เท่านั้น ห้ามส่งคลิปหรือไฟล์นี้ออกไปนอกเครื่อง (ข้อตกลง AVI6)",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="avi_eval_subset.csv")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--overlap", type=int, default=10, help="clips also labelled by the friend")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    keys = []
    with open(GROUND_TRUTH_DIR / args.manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys += row["clip_keys"].split()
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    keys = keys[:args.n]

    rows = []
    for i, key in enumerate(keys, 1):
        rows.append({"n": i, "key": key, "video": find_video(key),
                     "question": question_context(key)["text"],
                     "main_issue": "", "second_issue": "", "confidence (1-3)": "", "note": ""})
    missing = sum(r["video"] == "(not found)" for r in rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = write_sheet(REPORT_DIR / "issue_labels.csv", rows)
    friend = [dict(r) for r in rng.sample(rows, min(args.overlap, len(rows)))]
    friend.sort(key=lambda r: r["n"])
    p2 = write_sheet(REPORT_DIR / "issue_labels_friend.csv", friend)
    (REPORT_DIR / "issue_labels_GUIDE.md").write_text(guide_text(len(rows), len(friend)),
                                                     encoding="utf-8")
    print(f"[labels] {p1}  ({len(rows)} clips)")
    print(f"[labels] {p2}  ({len(friend)} clips for the second rater)")
    print(f"[labels] {REPORT_DIR / 'issue_labels_GUIDE.md'}")
    if missing:
        print(f"[WARN] {missing} video file(s) not found in input/videos_archive or input/videos")


if __name__ == "__main__":
    main()
