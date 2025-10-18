import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Assuming preprocess_data.py is run first to get tracking_data
from preprocess_data import load_tracking_data, load_annotations

def extract_pose_features(tracking_df):
    """Extract pose features from tracking data (keypoints)"""
    # Fixed list of bodyparts (12 in MABe dataset), fuse right_lateral and right_hip into 'right_hip_lateral'
    bodyparts = ['nose', 'left_ear', 'right_ear', 'neck', 'left_front_paw', 'right_front_paw',
                 'center', 'left_rear_paw', 'right_rear_paw', 'tail_base', 'tail_middle', 'tail_tip', 'right_hip_lateral']

    # Compute mean x,y per bodypart
    mean_coords = tracking_df.groupby('bodypart')[['x', 'y']].mean()

    # Fuse right_lateral and right_hip: if both present, average; else use whichever is present
    fused_coords = []
    if 'right_lateral' in mean_coords.index and 'right_hip' in mean_coords.index:
        fused_coords = (mean_coords.loc['right_lateral'] + mean_coords.loc['right_hip']) / 2
    elif 'right_lateral' in mean_coords.index:
        fused_coords = mean_coords.loc['right_lateral']
    elif 'right_hip' in mean_coords.index:
        fused_coords = mean_coords.loc['right_hip']
    else:
        fused_coords = pd.Series([0.0, 0.0], index=['x', 'y'])

    # Create feature vector with fixed size, pad missing bodyparts with 0
    features = []
    for bp in bodyparts[:-1]:  # Exclude fused one
        if bp in mean_coords.index:
            features.extend(mean_coords.loc[bp].values)
        else:
            features.extend([0.0, 0.0])  # Pad with zeros
    # Add fused
    features.extend(fused_coords.values)

    return np.array(features)

def prepare_unlabeled_data(tracking_data, annotations):
    """Prepare unlabeled data for unsupervised learning"""
    unlabeled_features = []
    unlabeled_keys = []
    for key, df in tracking_data.items():
        if key not in annotations or annotations[key].empty:
            features = extract_pose_features(df)
            unlabeled_features.append(features)
            unlabeled_keys.append(key)
    if unlabeled_features:
        # Concatenate all unlabeled features
        X_unlabeled = np.vstack(unlabeled_features)
        print(f"Prepared {len(X_unlabeled)} unlabeled samples from {len(unlabeled_keys)} videos")
        return X_unlabeled, unlabeled_keys
    else:
        return None, []

def unsupervised_learning(X_unlabeled, n_clusters=10):
    """Perform unsupervised learning on unlabeled data (e.g., clustering)"""
    if X_unlabeled is not None and len(X_unlabeled) >= n_clusters:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_unlabeled)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        print(f"Clustered unlabeled data into {n_clusters} clusters")
        return clusters, kmeans
    elif X_unlabeled is not None:
        print(f"Skipping clustering: only {len(X_unlabeled)} samples, need at least {n_clusters}")
    return None, None

def feature_engineering(video_ids=None):
    """Main feature engineering function"""
    tracking_data = load_tracking_data(video_ids)
    annotations = load_annotations()

    # Extract features for all data
    all_features = {}
    for key, df in tracking_data.items():
        features = extract_pose_features(df)
        all_features[key] = features

    # Prepare unlabeled data
    X_unlabeled, unlabeled_keys = prepare_unlabeled_data(tracking_data, annotations)

    # Perform unsupervised learning
    clusters, model = unsupervised_learning(X_unlabeled)

    # Integrate unsupervised features: add cluster distances to all features using the model
    if model is not None:
        scaler = StandardScaler()
        # Fit scaler on unlabeled data
        X_scaled_unlabeled = scaler.fit_transform(X_unlabeled)
        # Retrain model if needed, but since it's already fitted, use transform on all
        all_X = np.array(list(all_features.values()))
        X_scaled_all = scaler.transform(all_X)  # Scale all features
        distances_all = model.transform(X_scaled_all)  # Distances to centroids for all
        for i, key in enumerate(all_features.keys()):
            # Add distances to centroids as additional features
            all_features[key] = np.concatenate([all_features[key], distances_all[i]])

    return all_features, X_unlabeled, clusters, model

if __name__ == "__main__":
    features, X_unlabeled, clusters, model = feature_engineering()
    print("Feature engineering complete.")
