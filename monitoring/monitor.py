import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import os

def generate_monitoring_report():
    """Generate data drift report."""
    
    # Load reference and current data
    reference_data = pd.read_csv("dataprocessed/train.csv")
    current_data = pd.read_csv("dataprocessed/test.csv")
    
    print("Generating monitoring report...")
    
    # Create report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save report
    os.makedirs("monitoring/reports", exist_ok=True)
    report.save_html("monitoring/reports/drift_report.html")
    
    print("✅ Report saved to monitoring/reports/drift_report.html")

if __name__ == "__main__":
    generate_monitoring_report()