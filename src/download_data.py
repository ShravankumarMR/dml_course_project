import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def download_dataset(output_path= "dataraw/dataset.csv"):
    # Load the breast cancer dataset from sklearn
    data = load_breast_cancer()
    
    # Create a DataFrame with the features
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Add the target variable to the DataFrame
    df['target'] = data.target
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Dataset downloaded and saved to {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df
if __name__ == "__main__":
    download_dataset()
    print("Data download complete.")
