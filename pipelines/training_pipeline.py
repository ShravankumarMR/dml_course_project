"""
Prefect pipeline for ML workflow.
"""
from prefect import flow, task
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import preprocess_data
from train import train_models

@task(name="Preprocess Data")
def preprocess_task():
    """Preprocess data."""
    print(" Preprocessing data...")
    train_df, test_df = preprocess_data(
        input_path="dataraw/dataset.csv",
        output_dir="dataprocessed"
    )
    return True

@task(name="Train Models")
def train_task():
    """Train models."""
    print(" Training models...")
    best_model, best_name = train_models(
        train_path="dataprocessed/train.csv",
        test_path="dataprocessed/test.csv"
    )
    return best_name

@flow(name="ML Training Pipeline")
def ml_pipeline():
    """Complete ML pipeline."""
    print(" Starting ML Pipeline...")
    
    # Step 1: Preprocess
    preprocess_done = preprocess_task()
    
    # Step 2: Train
    best_model_name = train_task()
    
    print(f"\n Pipeline complete! Best model: {best_model_name}")
    return best_model_name

if __name__ == "__main__":
    result = ml_pipeline()
