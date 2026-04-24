"""
Configuration management for ML bimetallic catalyst design.
Pydantic v2 based configuration with validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class SlabConfig(BaseModel):
    """Slab geometry configuration."""
    n_layers: int = Field(4, description="Number of atomic layers in slab")
    supercell: tuple = Field((4, 4, 1), description="Supercell dimensions")
    vacuum: float = Field(15.0, description="Vacuum spacing in Angstrom")
    atomic_layer_height: float = Field(8.0, description="Height of atomic layers in Angstrom")
    top_layer_metals: int = Field(2, description="Number of top layers to substitute")
    min_lateral_dimension: float = Field(8.0, description="Minimum lateral dimension in Angstrom")


class FilteringConfig(BaseModel):
    """Energy and physical validity filtering configuration."""
    e_min: float = Field(-5.0, description="Minimum adsorption energy threshold (eV)")
    e_max: float = Field(0.5, description="Maximum adsorption energy threshold (eV)")
    confidence_threshold: float = Field(0.8, description="Model prediction confidence threshold")
    lattice_mismatch_threshold: float = Field(20.0, description="Maximum lattice mismatch percentage")
    max_strain: float = Field(0.15, description="Maximum allowed structural strain")


class MLConfig(BaseModel):
    """Machine learning model configuration."""
    random_state: int = Field(42, description="Random seed for reproducibility")
    test_size: float = Field(0.2, description="Test set fraction")
    cv_folds: int = Field(5, description="Cross-validation folds")
    
    # XGBoost hyperparameters
    xgb_max_depth: int = Field(6, description="XGBoost max tree depth")
    xgb_learning_rate: float = Field(0.1, description="XGBoost learning rate")
    xgb_n_estimators: int = Field(100, description="Number of boosting rounds")
    xgb_subsample: float = Field(0.8, description="Fraction of samples for training each tree")
    xgb_colsample_bytree: float = Field(0.8, description="Fraction of features for training each tree")
    
    # Random Forest hyperparameters
    rf_max_depth: Optional[int] = Field(None, description="Random Forest max depth")
    rf_n_estimators: int = Field(100, description="Number of trees in forest")
    rf_min_samples_split: int = Field(2, description="Minimum samples to split a node")
    
    # Stage 2 specific
    ratio_grid_size: int = Field(11, description="Number of ratio points (0.0 to 1.0 at 0.1 intervals)")


class DataConfig(BaseModel):
    """Data paths and settings."""
    raw_data_dir: str = Field("data/raw", description="Raw OCP predictions directory")
    processed_data_dir: str = Field("data/processed", description="Processed data directory")
    structures_dir: str = Field("data/structures", description="Generated structures directory")
    predictions_dir: str = Field("data/predictions", description="Model predictions directory")
    models_dir: str = Field("models", description="Trained models directory")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")
    log_file: str = Field("logs/experiment.log", description="Log file path")
    enable_file_logging: bool = Field(True, description="Enable file logging")
    enable_console_logging: bool = Field(True, description="Enable console logging")


class Config(BaseModel):
    """Main configuration class."""
    slab: SlabConfig = Field(default_factory=SlabConfig)
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    model: MLConfig = Field(default_factory=MLConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    class Config:
        validate_assignment = True
    
    def to_dict(self):
        """Convert config to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary."""
        return cls(**data)


# Global config instance
_default_config = None

def get_config() -> Config:
    """Get or create default config instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_config(config: Config) -> None:
    """Set global config instance."""
    global _default_config
    _default_config = config
