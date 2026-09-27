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

def mock_evaluation_run():
    print("=" * 60)
    print("  VALIDATION METRICS EVALUATION")
    print("=" * 60)
    
    # 1. ICC Example (Checking if humans agree)
    print("1. Inter-Rater Reliability (ICC)")
    # Mock data: 3 raters scoring 5 videos (1-5 scale)
    data = {
        'video_id': ['v1','v1','v1', 'v2','v2','v2', 'v3','v3','v3', 'v4','v4','v4', 'v5','v5','v5'],
        'rater_id': ['r1','r2','r3', 'r1','r2','r3', 'r1','r2','r3', 'r1','r2','r3', 'r1','r2','r3'],
        'score':    [4, 4, 5,       2, 2, 1,       3, 4, 3,       5, 5, 5,       1, 2, 2]
    }
    df = pd.DataFrame(data)
    icc_res = calculate_icc(df)
    
    print(icc_res)
    # Get ICC2 (single random raters)
    icc_value = icc_res['ICC'].iloc[1] # ICC2 is typically the second row
    print(f"   Human ICC Agreement: {icc_value:.3f} (Values > 0.75 are excellent)\n")
    
    # 2. Compare AI vs Human Average
    print("2. System vs Human Baseline")
    # Human average scores rounded to nearest band
    human_avg = [4, 2, 3, 5, 2]
    ai_scores = [4, 1, 3, 5, 2] # AI is slightly harsher on video 2
    
    qwk = calculate_qwk(human_avg, ai_scores)
    spearman = calculate_spearman(human_avg, ai_scores)
    
    print(f"   Quadratic Weighted Kappa (QWK): {qwk:.3f} (Values > 0.6 are good)")
    print(f"   Spearman Rank Correlation     : {spearman:.3f}")
    print("=" * 60)
    print("Note: Replace mock data with actual CSV outputs when raters are done.")

if __name__ == "__main__":
    mock_evaluation_run()
