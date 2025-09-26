# utils.py
import pandas as pd

def add_age_bin(X):
    """
    Adds an 'age_bin' categorical column to the dataframe based on age ranges.
    """
    X = X.copy()
    if 'age' in X.columns:
        X['age_bin'] = pd.cut(
            X['age'],
            bins=[0, 25, 35, 50, 65, 120],
            labels=['<25', '25-34', '35-49', '50-64', '65+']
        )
    return X
