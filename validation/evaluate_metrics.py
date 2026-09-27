"""
validation/evaluate_metrics.py

Calculates key validation metrics for the AI Interview system:
1. ICC (Inter-Rater Reliability) - Do humans agree?
2. Spearman Correlation - Does AI rank correctly?
3. QWK (Quadratic Weighted Kappa) - Does AI give accurate absolute scores?

Required dependencies:
pip install pandas scikit-learn scipy pingouin
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
import pingouin as pg

def calculate_qwk(human_scores, ai_scores):
    """
    Quadratic Weighted Kappa (QWK)
    Measures agreement between two raters, penalizing larger differences heavier.
    Scores should ideally be discrete 1-5 bands.
    """
    return cohen_kappa_score(human_scores, ai_scores, weights='quadratic')

def calculate_spearman(human_scores, ai_scores):
    """
    Spearman Rank Correlation
    Measures if the AI ranks the candidates in the same order as humans.
    """
    correlation, p_value = spearmanr(human_scores, ai_scores)
    return correlation

def calculate_icc(df_ratings):
    """
    Intraclass Correlation Coefficient (ICC)
    Measures how much multiple human raters agree with each other.
    Expects a DataFrame with: 'video_id', 'rater_id', 'score'
    """
    # ICC2 is generally used for inter-rater reliability with random raters
    icc = pg.intraclass_corr(data=df_ratings, targets='video_id', raters='rater_id', ratings='score')
    return icc

def get_ai_scores():
    import json
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    scores_dir = project_root / "output" / "scores"
    
    band_to_num = {
        "Excellent": 5, "ดีมาก": 5,
        "Good": 4, "ดี": 4,
        "Fair": 3, "ปานกลาง": 3,
        "Needs Work": 2, "ควรปรับ": 2,
        "Priority": 1, "ควรปรับมาก": 1
    }
    
    ai_scores = {}
    for f in scores_dir.glob("*.json"):
        vid = f.stem
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            scores = {}
            for dim in ['eye_contact', 'head_pose', 'hand_gesture', 'facial_expression', 'answer_quality']:
                if dim in data['dimensions']:
                    scores[dim] = band_to_num.get(data['dimensions'][dim]['band']['label_en'], 3)
            if 'fusion' in data and 'overall_band' in data['fusion']:
                scores['overall'] = band_to_num.get(data['fusion']['overall_band']['label_en'], 3)
            ai_scores[vid] = scores
    return ai_scores

def run_evaluation():
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "input" / "ground_truth" / "human_ratings.csv"
    
    if not csv_path.exists():
        print("Ground truth CSV not found.")
        return
        
    df = pd.read_csv(csv_path)
    dimensions = ['eye_contact', 'head_pose', 'hand_gesture', 'facial_expression', 'answer_quality']
    
    print("=" * 60)
    print("  VALIDATION METRICS EVALUATION")
    print("=" * 60)
    print("1. Inter-Rater Reliability (ICC)")
    
    # Calculate ICC for each dimension
    for dim in dimensions:
        # Create a clean dataframe for pingouin: video_id, rater_id, score
        df_dim = df[['video_id', 'rater_id', dim]].rename(columns={dim: 'score'}).dropna()
        if not df_dim.empty and df_dim['score'].nunique() > 1:
            try:
                icc_res = calculate_icc(df_dim)
                icc_value = icc_res['ICC'].iloc[1] # ICC2
                print(f"   {dim.ljust(20)}: {icc_value:.3f}")
            except Exception as e:
                print(f"   {dim.ljust(20)}: Could not calculate (needs variance)")
        else:
            print(f"   {dim.ljust(20)}: Not enough data or variance")

    # Average human scores for ground truth
    df_mean = df.groupby('video_id')[dimensions].mean().round().astype(int)
    # Add overall human score
    df_mean['overall'] = df_mean[dimensions].mean(axis=1).round().astype(int)
    
    ai_scores_dict = get_ai_scores()
    
    print("\n2. System vs Human Baseline (QWK & Spearman)")
    dimensions_plus_overall = dimensions + ['overall']
    
    print(f"{'Dimension':<20} | {'QWK':<6} | {'Spearman':<8}")
    print("-" * 45)
    
    for dim in dimensions_plus_overall:
        human_list = []
        ai_list = []
        for vid in df_mean.index:
            if vid in ai_scores_dict and dim in ai_scores_dict[vid]:
                human_list.append(df_mean.loc[vid, dim])
                ai_list.append(ai_scores_dict[vid][dim])
        
        if len(human_list) > 1:
            try:
                qwk = calculate_qwk(human_list, ai_list)
            except Exception:
                qwk = float('nan')
            try:
                spearman = calculate_spearman(human_list, ai_list)
            except Exception:
                spearman = float('nan')
            
            print(f"{dim:<20} | {qwk:>6.3f} | {spearman:>8.3f}")
        else:
            print(f"{dim:<20} | N/A    | N/A")
            
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
