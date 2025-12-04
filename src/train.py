import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os

def train_models(train_path, test_path, target_col='target'):
    """Train multiple models with MLflow tracking."""
    
    # Set MLflow experiment
    mlflow.set_experiment("breast-cancer-classification")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"Data loaded: Train={len(X_train)}, Test={len(X_test)}")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    # Train each model
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        with mlflow.start_run(run_name=name):
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "model")
            
            print(f"   {name} Results:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"\n{'='*50}")
    print(f"   Best Model: {best_name}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"Saved to models/best_model.pkl")
    
    return best_model, best_name

if __name__ == "__main__":
    best_model, best_name = train_models(
        train_path="dataprocessed/train.csv",
        test_path="dataprocessed/test.csv"
    )
    print("\n Training complete!")
