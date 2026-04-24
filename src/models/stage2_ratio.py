"""
Stage 2: ML model for optimizing atomic ratios in bimetallic catalysts.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.models.base_model import BaseModel
from src.data_pipeline.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class Stage2RatioModel(BaseModel):
    """
    Stage 2: Optimize atomic ratios for top metal pairs from Stage 1.
    
    For each metal pair (A, B), optimize composition A_xB_{1-x}
    where x varies from 0 to 1 at 0.1 intervals.
    
    Inputs: Composition-dependent features + x ratio
    Outputs: Optimal ratio and predicted adsorption energy
    """
    
    def __init__(self, model_type: str = 'xgboost', config=None):
        """
        Initialize Stage 2 model.
        
        Args:
            model_type: 'xgboost' or 'random_forest'
            config: Configuration object
        """
        super().__init__(model_type=model_type, config=config)
        
        if model_type == 'xgboost' and not HAS_XGBOOST:
            logger.warning("XGBoost not available, falling back to RandomForest")
            self.model_type = 'random_forest'
        
        self.feature_engineer = FeatureEngineer(config)
    
    def _build_model(self):
        """Build the underlying ML model."""
        
        if self.model_type == 'xgboost' and HAS_XGBOOST:
            self.model = XGBRegressor(
                max_depth=self.config.model.xgb_max_depth,
                learning_rate=self.config.model.xgb_learning_rate,
                n_estimators=self.config.model.xgb_n_estimators,
                subsample=self.config.model.xgb_subsample,
                random_state=self.config.model.random_state,
                verbosity=0
            )
        else:
            self.model = RandomForestRegressor(
                max_depth=self.config.model.rf_max_depth,
                n_estimators=self.config.model.rf_n_estimators,
                random_state=self.config.model.random_state,
                n_jobs=-1
            )
    
    def train(self, X_train, y_train, cv_folds: int = 5, **kwargs):
        """
        Train Stage 2 model.
        
        Args:
            X_train: Training features (including x ratio values)
            y_train: Training targets (adsorption energies)
            cv_folds: Cross-validation folds
            **kwargs: Additional arguments
        """
        
        logger.info(f"Training Stage 2 ratio optimization model...")
        logger.info(f"Samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        # Build model
        self._build_model()
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model,
            X_train, y_train,
            cv=cv_folds,
            scoring='r2'
        )
        
        logger.info(f"CV R² scores: {cv_scores}")
        logger.info(f"CV R² mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info("Training complete!")
    
    def predict(self, X) -> np.ndarray:
        """
        Make ratio predictions.
        
        Args:
            X: Feature matrix with composition-dependent features
        
        Returns:
            Predicted adsorption energies
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_optimal_ratio(self,
                             metal_a: str,
                             metal_b: str,
                             return_all: bool = False) -> Dict:
        """
        Predict optimal atomic ratio for a metal pair.
        
        Args:
            metal_a: First metal symbol
            metal_b: Second metal symbol
            return_all: If True, return predictions for all x values
        
        Returns:
            Dictionary with optimal ratio and energy
        """
        
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Evaluate all ratio points
        x_values = np.linspace(0, 1, self.config.model.ratio_grid_size)
        predictions = []
        
        for x in x_values:
            # Get features for this ratio
            ratio_features = self.feature_engineer.create_ratio_features(
                metal_a, metal_b, x
            )
            
            if not ratio_features:
                logger.warning(f"Cannot create features for {metal_a}-{metal_b} at x={x}")
                continue
            
            X_ratio = np.array(list(ratio_features.values())).reshape(1, -1)
            pred_energy = self.predict(X_ratio)[0]
            
            predictions.append({
                'x': float(x),
                'composition': f"{metal_a}_{x:.1f}{metal_b}_{1-x:.1f}",
                'energy': float(pred_energy),
            })
        
        # Find optimal (closest to volcano optimum)
        predictions = sorted(
            predictions,
            key=lambda p: abs(p['energy'] - (-0.2))
        )
        
        optimal = predictions[0]
        
        result = {
            'metal_a': metal_a,
            'metal_b': metal_b,
            'composition': f"{metal_a}-{metal_b}",
            'optimal_ratio_x': optimal['x'],
            'optimal_ratio_composition': optimal['composition'],
            'optimal_energy_offset': optimal['energy'],
        }
        
        if return_all:
            result['all_ratios'] = predictions
        
        return result
    
    def predict_ratio_series(self,
                            metal_a: str,
                            metal_b: str) -> pd.DataFrame:
        """
        Get full ratio series predictions for a metal pair.
        
        Args:
            metal_a: First metal symbol
            metal_b: Second metal symbol
        
        Returns:
            DataFrame with ratio and energy predictions
        """
        
        x_values = np.linspace(0, 1, self.config.model.ratio_grid_size)
        results = []
        
        for x in x_values:
            ratio_features = self.feature_engineer.create_ratio_features(
                metal_a, metal_b, x
            )
            
            if not ratio_features:
                continue
            
            X_ratio = np.array(list(ratio_features.values())).reshape(1, -1)
            pred_energy = self.predict(X_ratio)[0]
            
            results.append({
                'metal_a': metal_a,
                'metal_b': metal_b,
                'x': x,
                '1_minus_x': 1 - x,
                'composition': f"{metal_a}_{x:.1f}{metal_b}_{1-x:.1f}",
                'predicted_energy': pred_energy,
            })
        
        return pd.DataFrame(results)
    
    def train_on_top_pairs(self,
                          top_pairs: List[Dict],
                          x_values: Optional[np.ndarray] = None) -> Dict:
        """
        Train on ratio variations of top metal pairs from Stage 1.
        
        Args:
            top_pairs: List of top compositions from Stage 1
            x_values: Ratio points to evaluate (default: 0.0-1.0 at 0.1)
        
        Returns:
            Dictionary of optimized compositions
        """
        
        if x_values is None:
            x_values = np.linspace(0, 1, self.config.model.ratio_grid_size)
        
        logger.info(f"Training Stage 2 on {len(top_pairs)} top pairs...")
        logger.info(f"Ratio points: {len(x_values)} (x from {x_values.min():.1f} to {x_values.max():.1f})")
        
        optimized = {}
        for pair in top_pairs:
            metal_a = pair.get('metal_a', pair.get('composition', 'Metal1').split('-')[0])
            metal_b = pair.get('metal_b', pair.get('composition', 'Metal1').split('-')[1])
            
            result = self.predict_optimal_ratio(metal_a, metal_b)
            optimized[f"{metal_a}-{metal_b}"] = result
            
            logger.info(f"  {metal_a}-{metal_b}: optimal x = {result['optimal_ratio_x']:.1f}")
        
        return optimized
