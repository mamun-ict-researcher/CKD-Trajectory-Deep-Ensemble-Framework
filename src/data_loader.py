import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.impute import SimpleImputer

def compute_ckd_epi(creatinine, age, sex, race=None):
    # Standard clinical CKD-EPI formula
    try: scr = float(creatinine)
    except: return np.nan
    if scr <= 0: return np.nan
    sex_str = str(sex).lower() if pd.notna(sex) else 'm'
    k, alpha, sex_coef = (0.7, -0.329, 1.018) if sex_str.startswith('f') else (0.9, -0.411, 1.0)
    race_coef = 1.159 if isinstance(race, str) and 'black' in race.lower() else 1.0
    return 141.0 * min(scr/k,1.0)**alpha * max(scr/k,1.0)**(-1.209) * (0.993**float(age)) * sex_coef * race_coef

def clean_and_prepare_data(df, target_col, patient_col, time_col):
    # Standardization: Enforce consistent schema and handling missing values
    df.columns = [str(c).lower().strip() for c in df.columns]
    target_col, patient_col, time_col = target_col.lower(), patient_col.lower(), time_col.lower()

    if target_col not in df.columns:
        creat_cols = [c for c in df.columns if 'creatinine' in c]
        if creat_cols:
            df[target_col] = df.apply(lambda r: compute_ckd_epi(r.get(creat_cols[0], np.nan), r.get('age', 60), r.get('gender', 'M')), axis=1)
        else:
            raise KeyError(f"Schema Error: Target '{target_col}' not found.")

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [target_col, patient_col, 'hadm_id', 'hospital_expire_flag', 'anchor_age']
    features = [c for c in num_cols if c not in exclude]
    
    # Impute missing clinical data points
    imp = SimpleImputer(strategy='median')
    df[features] = imp.fit_transform(df[features])
    df[target_col] = df[target_col].fillna(df[target_col].median())
    
    return df, features

def build_windows(df, pid_col, features, window, horizon, target_col, time_col):
    rows = []
    pid_col, target_col, time_col = pid_col.lower(), target_col.lower(), time_col.lower()
    for pid, sub in tqdm(df.groupby(pid_col), desc="Building sequences"):
        sub = sub.sort_values(time_col).reset_index(drop=True)
        if len(sub) < (window + horizon): continue
        # Pre-convert to NumPy to avoid KeyError and optimize speed
        feat_arr, targ_arr = sub[features].values, sub[target_col].values
        for i in range(len(sub) - window - horizon + 1):
            x = feat_arr[i : i + window]
            y = targ_arr[i + window : i + window + horizon]
            if not np.isnan(x).any() and not np.isnan(y).any():
                rows.append({'pid': pid, 'seq': x, 'y': y})
    return rows