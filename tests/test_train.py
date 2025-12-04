import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_model_file_created():
    """Test if model file is created after training."""
    from train import train_models
    
    train_models(
        train_path="dataprocessed/train.csv",
        test_path="dataprocessed/test.csv"
    )
    
    assert os.path.exists("models/best_model.pkl")

def test_model_can_predict():
    """Test if saved model can make predictions."""
    import joblib
    import pandas as pd
    
    model = joblib.load("models/best_model.pkl")
    
    # Load test data
    test_df = pd.read_csv("dataprocessed/test.csv")
    X_test = test_df.drop(columns=['target'])
    
    # Make prediction
    prediction = model.predict(X_test[:1])
    
    assert prediction is not None
    assert prediction[0] in [0, 1]
