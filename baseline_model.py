import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import LabelEncoder
import os

# Assuming feature_engineering.py is run to get features
from feature_engineering import feature_engineering
from preprocess_data import load_train_csv, load_annotations

def f_beta_metric(y_true, y_pred, beta=1.0):
    """Custom F Beta metric for MABe"""
    # Updated to micro average based on discussion 610685: averaging per action (sample), not per video
    # Overview misstatement; micro weights by number of actions, matching competition metric
    return fbeta_score(y_true, y_pred, beta=beta, average='micro')

def prepare_labeled_data(train_df, all_features, annotations):
    """Prepare labeled data for training"""
    # Merge train_df with features and annotations
    # This is simplified; need to match by video_id or similar
    X = []
    y = []
    groups = []  # for GroupKFold by mouse_id
    all_actions = set()
    for idx, row in train_df.iterrows():
        video_id = str(row['video_id'])  # Assuming column exists
        if video_id in all_features and video_id in annotations:
            ann_df = annotations[video_id]
            # Assuming action column
            label = ann_df['action'].mode()[0] if not ann_df.empty else None
            if label is not None:
                features = all_features[video_id]
                # features is already aggregated per video
                X.append(features)
                y.append(label)
                all_actions.add(label)
                groups.append(row['mouse1_id'])  # Use mouse1_id as group
    # Encode labels
    if y:
        le = LabelEncoder()
        le.fit(list(all_actions))
        y_encoded = le.transform(y)
        return np.array(X), np.array(y_encoded), np.array(groups), le
    else:
        return np.array(X), np.array(y), np.array(groups), None

def augment_data(X, y, groups):
    """Simple data augmentation: left-right flip for pose data"""
    X_aug = []
    y_aug = []
    groups_aug = []
    for i in range(len(X)):
        # Original
        X_aug.append(X[i])
        y_aug.append(y[i])
        groups_aug.append(groups[i])
        # Flipped: swap left and right keypoints
        # Assuming features: nose(0-1), left_ear(2-3), right_ear(4-5), neck(6-7), left_front_paw(8-9), right_front_paw(10-11),
        # center(12-13), left_rear_paw(14-15), right_rear_paw(16-17), tail_base(18-19), tail_middle(20-21), tail_tip(22-23), right_hip_lateral(24-25)
        # Swap: left_ear <-> right_ear, left_front_paw <-> right_front_paw, left_rear_paw <-> right_rear_paw
        flipped = X[i].copy()
        # Swap ears: 2-3 <-> 4-5
        flipped[2:4], flipped[4:6] = flipped[4:6], flipped[2:4]
        # Swap front paws: 8-9 <-> 10-11
        flipped[8:10], flipped[10:12] = flipped[10:12], flipped[8:10]
        # Swap rear paws: 14-15 <-> 16-17
        flipped[14:16], flipped[16:18] = flipped[16:18], flipped[14:16]
        # Note: right_hip_lateral is fused, assume symmetric
        X_aug.append(flipped)
        y_aug.append(y[i])  # Same label
        groups_aug.append(groups[i])  # Same group
    return np.array(X_aug), np.array(y_aug), np.array(groups_aug)

def train_baseline_model(X, y, groups, n_neighbors=5):
    """Train KNN with cross-validation and hyperparameter tuning"""
    kf = GroupKFold(n_splits=5)
    scores = []
    for train_idx, val_idx in kf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        score = f_beta_metric(y_val, y_pred)
        scores.append(score)
    avg_score = np.mean(scores)
    print(f"Average F Beta score with n_neighbors={n_neighbors}: {avg_score}")
    return avg_score

def hyperparameter_tune(X, y, groups):
    """Simple hyperparameter tuning for KNN n_neighbors"""
    best_score = 0
    best_n = 5
    for n in [3, 5, 7, 9]:
        score = train_baseline_model(X, y, groups, n_neighbors=n)
        if score > best_score:
            best_score = score
            best_n = n
    print(f"Best n_neighbors: {best_n}, Score: {best_score}")
    return best_n, best_score

def baseline_model(n_runs=1):
    """Main baseline model function, optionally run multiple times for robustness"""
    # Get train data first to filter videos
    train_df = load_train_csv()
    annotations = load_annotations()  # Load annotations

    # Filter to labeled videos and valid mouse_id
    labeled_df = train_df[train_df['behaviors_labeled'].notna() & train_df['mouse1_id'].notna()]
    labeled_video_ids = set(str(row['video_id']) for _, row in labeled_df.iterrows())

    # Get features only for labeled videos
    all_features, _, _, _ = feature_engineering(video_ids=labeled_video_ids)

    # Prepare data
    X, y, groups, le = prepare_labeled_data(labeled_df, all_features, annotations)

    # Augment data
    X_aug, y_aug, groups_aug = augment_data(X, y, groups)
    print(f"Augmented data: {len(X_aug)} samples (original {len(X)})")

    # Hyperparameter tuning
    best_n, best_score = hyperparameter_tune(X_aug, y_aug, groups_aug)

    # Train and evaluate with best hyperparams multiple times if requested
    scores = []
    for _ in range(n_runs):
        score = train_baseline_model(X_aug, y_aug, groups_aug, n_neighbors=best_n)
        scores.append(score)
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Final after {n_runs} runs with n_neighbors={best_n}: Average F Beta score: {avg_score}, Std: {std_score}")
    return avg_score, std_score

if __name__ == "__main__":
    score = baseline_model()
    print(f"Baseline model evaluation complete. Score: {score}")
