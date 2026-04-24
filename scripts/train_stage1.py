"""
Example training script for Stage 1 composition prediction.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import Config, get_logger
from src.data_pipeline import OCPDataProcessor, FeatureEngineer
from src.models import Stage1CompositionModel
from src.evaluation import HERMetrics

logger = get_logger(__name__)


def main():
    """Run Stage 1 training pipeline."""
    
    logger.info("=" * 60)
    logger.info("Stage 1: Bimetallic Composition Prediction")
    logger.info("=" * 60)
    
    # Load configuration
    config = Config()
    logger.info(f"Configuration loaded")
    
    # Step 1: Process OCP predictions
    logger.info("\n[Step 1] Processing OCP predictions...")
    ocp_processor = OCPDataProcessor(config)
    
    # Example: load predictions from file
    ocp_file = "data/ocp_predictions.csv"  # You need to provide this file
    
    try:
        df_processed, top_comps = ocp_processor.process_pipeline(
            ocp_file, top_n=20
        )
        logger.info(f"Processed {len(df_processed)} valid compositions")
        
        # Save processed data
        ocp_processor.save_processed_data(
            df_processed, 
            "data/processed/ocp_processed.csv"
        )
        
    except FileNotFoundError:
        logger.error(f"OCP predictions file not found: {ocp_file}")
        logger.info("Please provide OCP predictions in CSV format with columns:")
        logger.info("  - metal_a, metal_b")
        logger.info("  - h_adsorption_energy, water_adsorption_energy")
        logger.info("  - lattice_constant_a, lattice_constant_b")
        logger.info("  - prediction_confidence")
        return
    
    # Step 2: Feature engineering
    logger.info("\n[Step 2] Feature engineering...")
    engineer = FeatureEngineer(config)
    
    try:
        X_train, y_train, X_test, y_test = engineer.prepare_dataset(
            "data/processed/ocp_processed.csv",
            target_col='h_energy_offset',
            train_size=0.8
        )
        logger.info(f"Dataset prepared: {len(X_train)} train, {len(X_test)} test")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return
    
    # Step 3: Train Stage 1 model
    logger.info("\n[Step 3] Training Stage 1 model...")
    stage1 = Stage1CompositionModel(model_type='xgboost', config=config)
    
    stage1.train(
        X_train, y_train,
        cv_folds=config.model.cv_folds,
        hyperparameter_tuning=False  # Set to True for GridSearch
    )
    
    # Step 4: Evaluate model
    logger.info("\n[Step 4] Evaluating Stage 1 model...")
    metrics = stage1.evaluate(X_test, y_test)
    
    # Step 5: Rank compositions
    logger.info("\n[Step 5] Ranking top compositions...")
    top_rankings = stage1.rank_compositions(
        X_test,
        compositions=[(f"Metal_A", f"Metal_B")] * len(X_test),
        top_n=5
    )
    
    # Step 6: Feature importance
    logger.info("\n[Step 6] Feature importance analysis...")
    importance_df = stage1.get_feature_importance(engineer.feature_names)
    
    # Save model
    logger.info("\n[Step 7] Saving trained model...")
    stage1.save("models/stage1_composition.pkl")
    
    logger.info("\n" + "=" * 60)
    logger.info("Stage 1 Training Complete!")
    logger.info("=" * 60)
    
    # Summary
    logger.info("\nResults Summary:")
    logger.info(f"  Train samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Test R²: {metrics['r2']:.4f}")
    logger.info(f"  Test RMSE: {metrics['rmse']:.6f} eV")
    logger.info(f"  Top composition: {top_rankings[0] if top_rankings else 'N/A'}")
    
    return stage1, engineer, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    main()
