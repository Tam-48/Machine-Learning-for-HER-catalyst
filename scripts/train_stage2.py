"""
Example training script for Stage 2 ratio optimization.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import Config, get_logger
from src.models import Stage2RatioModel
from src.data_pipeline import FeatureEngineer

logger = get_logger(__name__)


def main():
    """Run Stage 2 training pipeline."""
    
    logger.info("=" * 60)
    logger.info("Stage 2: Atomic Ratio Optimization")
    logger.info("=" * 60)
    
    # Load configuration
    config = Config()
    logger.info(f"Configuration loaded")
    
    # Step 1: Prepare ratio dataset
    logger.info("\n[Step 1] Preparing ratio dataset...")
    
    # Generate synthetic ratio data for demonstration
    # In practice, this would come from OCP predictions for ratio variations
    
    x_values = np.linspace(0, 1, 11)  # 11 points: 0.0, 0.1, ..., 1.0
    
    # Create synthetic training data for demonstration
    n_samples = 200
    n_compositions = 5
    
    data = []
    for comp_idx in range(n_compositions):
        for x in x_values:
            for _ in range(n_samples // (n_compositions * len(x_values))):
                # Synthetic features (in practice from OCP)
                z_a = 26 + comp_idx * 5  # Varying atomic number
                z_b = 28
                en_a = 1.8 + comp_idx * 0.1
                en_b = 1.9
                
                # Synthetic target: volcano-shaped curve
                h_energy_offset = 0.3 * (x - 0.5)**2 + np.random.normal(0, 0.05)
                
                data.append({
                    'metal_a': f"Metal{comp_idx}",
                    'metal_b': "RefMetal",
                    'atomic_fraction_x': x,
                    'atomic_fraction_1mx': 1 - x,
                    'z_A': z_a,
                    'z_B': z_b,
                    'en_A': en_a,
                    'en_B': en_b,
                    'z_weighted': x * z_a + (1-x) * z_b,
                    'en_weighted': x * en_a + (1-x) * en_b,
                    'h_energy_offset': h_energy_offset
                })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} training samples")
    
    # Step 2: Feature engineering for ratio data
    logger.info("\n[Step 2] Feature selection...")
    
    feature_cols = ['z_A', 'z_B', 'en_A', 'en_B', 
                   'atomic_fraction_x', 'atomic_fraction_1mx',
                   'z_weighted', 'en_weighted']
    
    X = df[feature_cols]
    y = df['h_energy_offset']
    
    # Split dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.model.random_state
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 3: Train Stage 2 model
    logger.info("\n[Step 3] Training Stage 2 ratio model...")
    
    stage2 = Stage2RatioModel(model_type='xgboost', config=config)
    stage2.train(X_train, y_train, cv_folds=config.model.cv_folds)
    
    # Step 4: Evaluate model
    logger.info("\n[Step 4] Evaluating Stage 2 model...")
    metrics = stage2.evaluate(X_test, y_test)
    
    # Step 5: Predict optimal ratios
    logger.info("\n[Step 5] Predicting optimal ratios...")
    
    # For each top composition from Stage 1
    top_pairs = [
        {'metal_a': 'Metal0', 'metal_b': 'RefMetal'},
        {'metal_a': 'Metal1', 'metal_b': 'RefMetal'},
    ]
    
    optimized_results = []
    
    for pair in top_pairs:
        # Create base features for this pair
        pair_data = df[df['metal_a'] == pair['metal_a']].iloc[0:1]
        
        result = stage2.predict_optimal_ratio(
            pair_data,
            pair['metal_a'],
            pair['metal_b']
        )
        
        optimized_results.append(result)
    
    # Save model
    logger.info("\n[Step 6] Saving trained model...")
    stage2.save("models/stage2_ratio.pkl")
    
    # Save results
    results_df = pd.DataFrame({
        'composition': [r['composition'] for r in optimized_results],
        'optimal_ratio': [r['optimal_ratio_x'] for r in optimized_results],
        'optimal_energy': [r['optimal_energy_offset'] for r in optimized_results]
    })
    
    results_df.to_csv("data/predictions/optimal_ratios.csv", index=False)
    logger.info(f"Results saved to data/predictions/optimal_ratios.csv")
    
    logger.info("\n" + "=" * 60)
    logger.info("Stage 2 Training Complete!")
    logger.info("=" * 60)
    
    # Summary
    logger.info("\nResults Summary:")
    logger.info(f"  Train samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Test R²: {metrics['r2']:.4f}")
    logger.info(f"  Test RMSE: {metrics['rmse']:.6f} eV")
    logger.info(f"\nOptimal compositions found:")
    for result in optimized_results:
        logger.info(f"  {result['composition']}: "
                   f"x={result['optimal_ratio_x']:.1f}, "
                   f"ΔE={result['optimal_energy_offset']:.3f} eV")
    
    return stage2


if __name__ == "__main__":
    main()
