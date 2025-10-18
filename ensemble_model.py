import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os

# Assuming feature_engineering.py and preprocess_data.py are run first
from feature_engineering import feature_engineering
from preprocess_data import load_train_csv, load_annotations
from baseline_model import f_beta_metric, prepare_labeled_data, augment_data

def train_binary_xgboost(X_train, y_train, behavior):
    """Train a binary XGBoost classifier for a specific behavior"""
    # Binary labels: 1 if behavior, 0 otherwise
    y_binary = (y_train == behavior).astype(int)

    # Hyperparameters (can be tuned)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    dtrain = xgb.DMatrix(X_train, label=y_binary)
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

def predict_binary_xgboost(model, X):
    """Predict probabilities for binary XGBoost"""
    dtest = xgb.DMatrix(X)
    probs = model.predict(dtest)
    return probs

def ensemble_predict(X, models, behaviors, threshold=0.5):
    """Combine predictions from binary classifiers"""
    predictions = []
    for i in range(len(X)):
        probs = {}
        for behavior, model in zip(behaviors, models):
            prob = predict_binary_xgboost(model, X[i:i+1])[0]
            probs[behavior] = prob

        # Select behavior with highest prob > threshold, else 'other'
        max_prob = max(probs.values())
        if max_prob > threshold:
            pred = max(probs, key=probs.get)
        else:
            pred = 'other'
        predictions.append(pred)
    return np.array(predictions)

def train_ensemble_model(X, y, groups, behaviors):
    """Train ensemble of binary XGBoost classifiers"""
    kf = GroupKFold(n_splits=5)
    scores = []
    models_list = []

    for train_idx, val_idx in kf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train binary models for each behavior
        models = []
        for behavior in behaviors:
            model = train_binary_xgboost(X_train, y_train, behavior)
            models.append(model)

        # Predict on validation
        y_pred = ensemble_predict(X_val, models, behaviors, threshold=0.0)
        score = f_beta_metric(y_val, y_pred)
        scores.append(score)
        models_list.append(models)  # Store models for each fold

    avg_score = np.mean(scores)
    print(f"Average F Beta score for ensemble: {avg_score}")
    return avg_score, models_list

def ensemble_model(n_runs=1):
    """Main ensemble model function"""
    # Get train data
    train_df = load_train_csv()
    annotations = load_annotations()

    # Filter to labeled videos
    labeled_df = train_df[train_df['behaviors_labeled'].notna() & train_df['mouse1_id'].notna()]
    labeled_video_ids = set(str(row['video_id']) for _, row in labeled_df.iterrows())

    # Get features
    all_features, _, _, _ = feature_engineering(video_ids=labeled_video_ids)

    # Prepare data
    X, y, groups, le = prepare_labeled_data(labeled_df, all_features, annotations)

    # Augment data
    X_aug, y_aug, groups_aug = augment_data(X, y, groups)
    print(f"Augmented data: {len(X_aug)} samples")

    # Get unique behaviors
    behaviors = np.unique(y_aug)
    print(f"Behaviors: {behaviors}")

    # Train and evaluate
    scores = []
    for _ in range(n_runs):
        score, models_list = train_ensemble_model(X_aug, y_aug, groups_aug, behaviors)
        scores.append(score)
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Final ensemble after {n_runs} runs: Average F Beta score: {avg_score}, Std: {std_score}")
    return avg_score, std_score

if __name__ == "__main__":
    score = ensemble_model()
    print(f"Ensemble model evaluation complete. Score: {score}")
