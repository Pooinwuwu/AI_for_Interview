import json
import csv
from pathlib import Path

def generate_template():
    project_root = Path(__file__).resolve().parents[1]
    metadata_path = project_root / "input" / "metadata" / "metadata.json"
    output_csv = project_root / "input" / "ground_truth" / "human_ratings.csv"
    
    # Ensure directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Load videos
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    videos = sorted(metadata.keys())
    raters = ["Rater_1", "Rater_2", "Rater_3"]
    
    # Define columns based on the 5 dimensions
    fieldnames = [
        "video_id", 
        "rater_id", 
        "eye_contact", 
        "head_pose", 
        "hand_gesture", 
        "facial_expression", 
        "answer_quality", 
        "notes"
    ]
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for vid in videos:
            for rater in raters:
                writer.writerow({
                    "video_id": vid,
                    "rater_id": rater,
                    "eye_contact": "",
                    "head_pose": "",
                    "hand_gesture": "",
                    "facial_expression": "",
                    "answer_quality": "",
                    "notes": ""
                })
                
    print(f"Generated ground truth template with {len(videos)} videos and {len(raters)} raters per video.")
    print(f"Saved to: {output_csv}")

if __name__ == "__main__":
    generate_template()
