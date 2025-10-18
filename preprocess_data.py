import pandas as pd
import numpy as np
import os
import glob

# Paths (adjust if necessary)
DATA_DIR = '../MABEDatasets/MABe-extracted'  # Dataset is in this directory
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TRACKING_DIR = os.path.join(DATA_DIR, 'train_tracking')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'train_annotation')

def load_train_csv():
    """Load train.csv"""
    if os.path.exists(TRAIN_CSV):
        df = pd.read_csv(TRAIN_CSV)
        print(f"Loaded train.csv with {len(df)} rows")
        return df
    else:
        print("train.csv not found")
        return None

def load_tracking_data(video_ids=None):
    """Load tracking parquet files, optionally filtered by video_ids"""
    tracking_files = glob.glob(os.path.join(TRACKING_DIR, '**', '*.parquet'), recursive=True)
    tracking_data = {}
    for file in tracking_files:
        key = os.path.basename(file).replace('.parquet', '')
        if video_ids is not None and key not in video_ids:
            continue
        df = pd.read_parquet(file)
        tracking_data[key] = df
        print(f"Loaded tracking for {key}: {len(df)} rows")
    return tracking_data

def load_annotations():
    """Load annotation parquet files"""
    annotation_files = glob.glob(os.path.join(ANNOTATION_DIR, '**', '*.parquet'), recursive=True)
    annotations = {}
    for file in annotation_files:
        key = os.path.basename(file).replace('.parquet', '')
        df = pd.read_parquet(file)
        annotations[key] = df
        print(f"Loaded annotations for {key}: {len(df)} rows")
    return annotations

def correct_arena_size(tracking_df, original_size_cm=50, corrected_size_cm=30, arena_pixels=800):
    """Correct arena size and rescale tracking data"""
    # Original pix/cm = arena_pixels / original_size_cm
    # Corrected pix/cm = arena_pixels / corrected_size_cm
    scale_factor = corrected_size_cm / original_size_cm
    # Rescale coordinates (assuming x, y columns)
    if 'x' in tracking_df.columns and 'y' in tracking_df.columns:
        tracking_df['x'] *= scale_factor
        tracking_df['y'] *= scale_factor
    return tracking_df

def handle_data_inconsistencies(tracking_df, train_row):
    """Handle data inconsistencies from discussion 609092"""
    # Trust fps over duration for tracked span
    fps = train_row.get('fps', 30)  # Default 30 if missing
    duration = train_row.get('duration', 0)
    expected_frames = int(fps * duration)
    actual_frames = len(tracking_df) if 'frame' in tracking_df.columns else len(tracking_df)
    # If disagreement, trust fps; no filtering here, just note

    # Clip coordinates outside bounds (assume 800x800 arena)
    if 'x' in tracking_df.columns and 'y' in tracking_df.columns:
        tracking_df['x'] = tracking_df['x'].clip(0, 800)
        tracking_df['y'] = tracking_df['y'].clip(0, 800)

    return tracking_df

def filter_short_actions(annotations_df, fps=30, min_duration=0.1):
    """Filter out very short actions ≤0.1s from discussion 609092"""
    if annotations_df.empty or 'start_frame' not in annotations_df.columns or 'end_frame' not in annotations_df.columns:
        return annotations_df
    # Calculate duration in seconds
    annotations_df['duration'] = (annotations_df['end_frame'] - annotations_df['start_frame']) / fps
    # Filter out short actions
    filtered = annotations_df[annotations_df['duration'] > min_duration].copy()
    return filtered

def preprocess_data():
    """Main preprocessing function"""
    train_df = load_train_csv()
    tracking_data = load_tracking_data()
    annotations = load_annotations()

    # Correct arena size and handle inconsistencies for all tracking data
    for key, df in tracking_data.items():
        tracking_data[key] = correct_arena_size(df)
        # Get corresponding train row for inconsistencies
        train_row = train_df[train_df['video_id'] == int(key)].iloc[0] if not train_df.empty else {}
        tracking_data[key] = handle_data_inconsistencies(tracking_data[key], train_row)
        print(f"Corrected arena size and handled inconsistencies for {key}")

    # Filter short actions in annotations
    for key, df in annotations.items():
        annotations[key] = filter_short_actions(df)
        print(f"Filtered short actions for {key}")

    # Handle missing annotations: Note which videos have no annotations
    missing_annotations = []
    for key in tracking_data.keys():
        if key not in annotations:
            missing_annotations.append(key)
            print(f"No annotations for {key} - can be used for unsupervised learning")

    # Save preprocessed data (optional)
    # For now, just return the data
    return train_df, tracking_data, annotations, missing_annotations

if __name__ == "__main__":
    train_df, tracking_data, annotations, missing = preprocess_data()
    print("Preprocessing complete.")
