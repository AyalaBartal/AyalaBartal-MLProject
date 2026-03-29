#!/usr/bin/env python3
"""Train Random Forest model using the RF trainer module"""

import os
import pandas as pd
from src.specific.rf.trainer.pe_rf_trainer_provider import RfPeTrainerProvider
from src.specific.rf.trainer.pe_rf_train_algo_args import RfPeTrainAlgoArgs
from src.specific.rf.trainer.pe_rf_train_report_args import RfPeTrainReportArgs

def main():
    print("=" * 60)
    print("RANDOM FOREST MODEL TRAINING")
    print("=" * 60)
    
    # Define paths
    goodware_csv = "uploads/goodware.csv"
    malware_csv = "uploads/brazilian-malware.csv"
    models_dir = "models/random_forest"
    combined_csv = "uploads/combined_pe_data.csv"
    
    # Create output directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Combine datasets if not already done
    if not os.path.exists(combined_csv):
        print("\nCombining goodware and malware datasets...")
        df_goodware = pd.read_csv(goodware_csv)
        df_goodware['Label'] = 0
        
        df_malware = pd.read_csv(malware_csv)
        df_malware['Label'] = 1
        
        df_combined = pd.concat([df_goodware, df_malware], ignore_index=True)
        df_combined.to_csv(combined_csv, index=False)
        print(f"✓ Combined dataset created: {len(df_combined)} samples")
    
    # Create training arguments
    algo_args = RfPeTrainAlgoArgs()
    report_args = RfPeTrainReportArgs(
        input_csv=combined_csv,
        out_report_dir=models_dir
    )
    
    # Get the complete training pipeline
    io_trainer = RfPeTrainerProvider.get_io_trainer()
    
    # Run training
    print("\nStarting model training...")
    io_trainer.train(algo_args, report_args)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Model saved to: {report_args.out_model_joblib}")
    print(f"✓ Metrics saved to: {report_args.out_report_json}")
    print(f"✓ Feature schema saved to: {report_args.out_schema_json}")
    print(f"✓ Feature importance saved to: {report_args.feature_importance_csv}")

if __name__ == '__main__':
    main()
