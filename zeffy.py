
# Zeffy: Advanced AutoML Pipeline 
# -----------------------------------------------------
# This file is a consolidated version of the Zeffy AutoML pipeline.
# For detailed documentation, please refer to the accompanying README.md file.
# -----------------------------------------------------
import os
import sys
import yaml
import json
import logging
import random
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Type, Literal, Callable
import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, Field, validator, root_validator, Extra
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    class nn:
        Module = object; Linear = object; ReLU = object; Sequential = object; BCEWithLogitsLoss = object; CrossEntropyLoss = object; MSELoss = object; BatchNorm1d=object; Dropout=object
    class optim:
        Adam = object; SGD = object
    class TensorDataset: pass
    class DataLoader: pass
    class torch:
        device = lambda x: x; tensor = lambda x, **k: x; float32=None; long=None; manual_seed=lambda x:x; sigmoid=lambda x:x; softmax=lambda x,**k:x; argmax=lambda x,**k:x; no_grad=lambda : type('no_grad', (), {'__enter__': lambda: None, '__exit__': lambda w,x,y,z: None})
        cuda = type('cuda', (), {'is_available': lambda: False, 'manual_seed_all': lambda x:x})()

import joblib
# --- Zeffy Code Starts Here ---


# --- Consolidated Zeffy Modules --- 

# --- Zeffy Exceptions --- 
# zeffy/utils/exceptions.py

class ZeffyError(Exception):
    """Base class for exceptions in Zeffy."""
    pass

class ZeffyConfigurationError(ZeffyError):
    """Raised for errors in pipeline configuration."""
    pass

class ZeffyDataError(ZeffyError):
    """Raised for errors related to data loading, validation, or processing."""
    pass

class ZeffyModelError(ZeffyError):
    """Raised for errors related to model initialization, training, or prediction."""
    pass

class ZeffyPreprocessingError(ZeffyError):
    """Raised for errors during data preprocessing."""
    pass

class ZeffyFeatureEngineeringError(ZeffyError):
    """Raised for errors during feature engineering."""
    pass

class ZeffyTuningError(ZeffyError):
    """Raised for errors during hyperparameter tuning."""
    pass

class ZeffyEnsemblingError(ZeffyError):
    """Raised for errors during model ensembling."""
    pass

class ZeffyEvaluationError(ZeffyError):
    """Raised for errors during model evaluation."""
    pass

class ZeffyExplainabilityError(ZeffyError):
    """Raised for errors during model explainability tasks."""
    pass

class ZeffyNotFittedError(ZeffyModelError):
    """Raised when attempting to use a model component that has not been fitted yet."""
    def __init__(self, message="This Zeffy component has not been fitted yet. Please call fit() before using this method."):
        super().__init__(message)


# --- Zeffy Config Models --- 
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator

class GlobalSettings(BaseModel):
    random_seed: int = Field(42, description="Global random seed for reproducibility.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field("INFO", description="Logging level.")
    results_path: str = Field("results/zeffy_run", description="Path to save results, models, and logs.")
    cache_intermediate_results: bool = Field(True, description="Whether to cache intermediate results to speed up reruns.")

class DataLoaderConfig(BaseModel):
    type: str = Field("csv", description="Type of data source (e.g., csv, parquet, database).")
    path: Optional[str] = Field(None, description="Path to the data file (if applicable).")
    target_column: str = Field("target", description="Name of the target variable column.")
    # Add other data loading params like separator, sheet_name for excel, db_connection_string etc.
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Extra parameters for the data loader.")

class PreprocessingStepConfig(BaseModel):
    name: str = Field(..., description="Name of the preprocessing step (e.g., missing_imputer, one_hot_encoder).")
    columns: Optional[Union[List[str], Literal["all", "numeric", "categorical"]]] = Field(None, description="Columns to apply this step to.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the preprocessing step.")

class FeatureEngineeringConfig(BaseModel):
    automated_feature_synthesis: bool = Field(False, description="Enable automated feature synthesis (e.g., DeepFeatureSynthesis).")
    polynomial_features: Optional[Dict[str, Any]] = Field(None, description="Configuration for polynomial features.")
    interaction_features: Optional[Dict[str, Any]] = Field(None, description="Configuration for interaction features.")
    feature_selection: Optional[Dict[str, Any]] = Field(None, description="Configuration for feature selection (e.g., RFE, SHAP-based).")
    custom_generators: List[Dict[str, Any]] = Field(default_factory=list, description="List of custom feature generator configurations.")

class ModelConfig(BaseModel):
    type: str = Field(..., description="Type of the model (e.g., lightgbm, xgboost, catboost, sklearn_logistic, pytorch_nn).")
    task_type: Optional[Literal["classification", "regression"]] = Field(None, description="Explicitly set task type, otherwise inferred.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters for the model.")
    # For NN/Pretrained models
    architecture_definition: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Path to model architecture file or dict defining it.")
    pretrained_model_path: Optional[str] = Field(None, description="Path to a pretrained model checkpoint.")
    fine_tune: bool = Field(False, description="Whether to fine-tune the pretrained model.")

class TuningConfig(BaseModel):
    enabled: bool = Field(True, description="Enable hyperparameter tuning.")
    optimizer: Literal["optuna", "hyperopt", "grid_search", "random_search"] = Field("optuna", description="HPO library/optimizer to use.")
    n_trials: int = Field(50, description="Number of trials for HPO.")
    timeout_per_trial: Optional[int] = Field(None, description="Timeout in seconds for each HPO trial.")
    metric_to_optimize: str = Field(..., description="Metric to optimize during HPO (e.g., roc_auc, neg_mean_squared_error).")
    direction: Literal["maximize", "minimize"] = Field("maximize", description="Direction of optimization for the metric.")
    search_space: Optional[Dict[str, Any]] = Field(None, description="Custom search space definition, otherwise use model defaults.")

class EnsemblingConfig(BaseModel):
    enabled: bool = Field(True, description="Enable model ensembling.")
    method: Literal["averaging", "weighted_averaging", "median", "stacking", "blending"] = Field("stacking", description="Ensembling method.")
    stacking_meta_learner: Optional[ModelConfig] = Field(None, description="Configuration for the meta-learner in stacking.")
    # Add other ensembling params like weights for weighted_averaging, blend_ratio etc.
    final_model_selection_strategy: Literal["best_single_model", "ensemble"] = Field("ensemble", description="Strategy for final model selection.")

class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(default_factory=lambda: ["roc_auc", "accuracy", "f1"], description="List of metrics for evaluation.")
    cross_validation_strategy: Dict[str, Any] = Field(default_factory=lambda: {"type": "StratifiedKFold", "n_splits": 5}, description="Cross-validation strategy.")

class ExplainabilityConfig(BaseModel):
    enabled: bool = Field(True, description="Enable model explainability features.")
    method: Literal["shap", "lime"] = Field("shap", description="Explainability method to use.")
    shap_explainer_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the SHAP explainer.")

class ZeffyConfig(BaseModel):
    project_name: str = Field("Zeffy AutoML Project", description="Name of the project.")
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    data_loader: DataLoaderConfig
    preprocessing: List[PreprocessingStepConfig] = Field(default_factory=list)
    feature_engineering: Optional[FeatureEngineeringConfig] = Field(default_factory=FeatureEngineeringConfig)
    models: List[ModelConfig] # Allow multiple models for comparison or ensembling
    tuning: Optional[TuningConfig] = None # Make tuning optional
    ensembling: Optional[EnsemblingConfig] = None # Make ensembling optional
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    explainability: Optional[ExplainabilityConfig] = Field(default_factory=ExplainabilityConfig)

    @validator("models")
    def check_models_not_empty(cls, v):
        if not v:
            raise ValueError("Models list cannot be empty.")
        return v

    class Config:
        validate_assignment = True
        extra = "forbid" # Forbid extra fields not defined in the model

# Example of how to use it (for testing, will be in loader.py)
# if __name__ == "__main__": # Zeffy: Main block from module, commented out.
    sample_config_dict = {
        "project_name": "Credit Scoring AutoML",
        "global_settings": {
            "random_seed": 123,
            "log_level": "DEBUG",
            "results_path": "outputs/credit_scoring"
        },
        "data_loader": {
            "type": "csv",
            "path": "data/train.csv",
            "target_column": "default_status"
        },
        "preprocessing": [
            {"name": "missing_imputer", "columns": "numeric", "params": {"strategy": "median"}},
            {"name": "one_hot_encoder", "columns": "categorical", "params": {"handle_unknown": "ignore"}}
        ],
        "feature_engineering": {
            "polynomial_features": {"degree": 2, "interaction_only": True, "columns": ["age", "income"]}
        },
        "models": [
            {"type": "lightgbm", "task_type": "classification", "params": {"n_estimators": 200, "learning_rate": 0.05}},
            {"type": "xgboost", "task_type": "classification", "params": {"n_estimators": 150}}
        ],
        "tuning": {
            "enabled": True,
            "optimizer": "optuna",
            "n_trials": 100,
            "metric_to_optimize": "roc_auc",
            "direction": "maximize"
        },
        "ensembling": {
            "enabled": True,
            "method": "stacking",
            "stacking_meta_learner": {"type": "sklearn_logistic", "task_type": "classification"}
        },
        "evaluation": {
            "metrics": ["roc_auc", "accuracy", "precision", "recall", "f1"],
            "cross_validation_strategy": {"type": "StratifiedKFold", "n_splits": 10, "shuffle": True}
        },
        "explainability": {
            "enabled": True,
            "method": "shap"
        }
    }

    try:
        zeffy_config = ZeffyConfig(**sample_config_dict)
        print("Config parsed successfully!")
        print(f"Project Name: {zeffy_config.project_name}")
        print(f"Random Seed: {zeffy_config.global_settings.random_seed}")
        print(f"First model type: {zeffy_config.models[0].type}")
        if zeffy_config.tuning:
            print(f"Tuning enabled: {zeffy_config.tuning.enabled}")
    except Exception as e: # PydanticValidationError
        print(f"Config validation error: {e}")


# --- Zeffy Config Loader --- 
import yaml
import json
from pathlib import Path
from typing import Dict, Any
# from .models import ZeffyConfig # Zeffy: Relative import handled by consolidation order
from ..utils.exceptions import ZeffyConfigurationError, ZeffyDataError

def load_config_from_file(config_path: str) -> Dict[Any, Any]:
    """Loads configuration from a YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise ZeffyConfigurationError(f"Configuration file not found: {config_path}")

    try:
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "r") as f:
                config_data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                config_data = json.load(f)
        else:
            raise ZeffyConfigurationError(f"Unsupported configuration file format: {path.suffix}. Please use YAML or JSON.")
        return config_data if config_data is not None else {}
    except Exception as e:
        raise ZeffyConfigurationError(f"Error reading configuration file {config_path}: {e}")

def load_config(config_path: str = None, config_dict: Dict[Any, Any] = None) -> ZeffyConfig:
    """Loads and validates the pipeline configuration from a file or dictionary.

    Args:
        config_path (str, optional): Path to a YAML or JSON configuration file.
        config_dict (dict, optional): A dictionary containing the configuration.

    Returns:
        ZeffyConfig: A validated Pydantic model instance of the configuration.

    Raises:
        ZeffyConfigurationError: If neither config_path nor config_dict is provided, or if config is invalid.
    """
    raw_config: Dict[Any, Any]
    if config_path:
        raw_config = load_config_from_file(config_path)
    elif config_dict:
        raw_config = config_dict
    else:
        # This will likely fail validation in ZeffyConfig if required fields are missing.
        print("Warning: No configuration path or dictionary provided. Attempting to initialize with an empty configuration.")
        raw_config = {} 

    try:
        validated_config = ZeffyConfig(**raw_config)
        return validated_config
    except Exception as e: # PydanticValidationError is an Exception subclass
        error_message = f"Zeffy configuration validation failed: {e}. Please check your configuration structure and values."
        # For Pydantic V2, e.errors() gives more detailed error list
        # detailed_errors = e.errors() if hasattr(e, 'errors') else str(e)
        # print(f"Detailed Pydantic errors: {detailed_errors}") # For debugging
        raise ZeffyConfigurationError(error_message)

# if __name__ == "__main__": # Zeffy: Main block from module, commented out.
    # Create dummy config files for testing loader.py directly
    import os
    dummy_configs_dir = "/home/ubuntu/zeffy_loader_test_configs"
    if not os.path.exists(dummy_configs_dir):
        os.makedirs(dummy_configs_dir)

    dummy_yaml_path = os.path.join(dummy_configs_dir, "test_config.yaml")
    dummy_json_path = os.path.join(dummy_configs_dir, "test_config.json")

    sample_data_for_loader = {
        "project_name": "Loader Test Project",
        "data_loader": {
            "type": "csv",
            "path": "/data/input.csv",
            "target_column": "outcome"
        },
        "models": [
            {"type": "lightgbm", "task_type": "classification", "params": {"n_estimators": 50}}
        ]
    }

    with open(dummy_yaml_path, 'w') as f:
        yaml.dump(sample_data_for_loader, f)

    with open(dummy_json_path, 'w') as f:
        json.dump(sample_data_for_loader, f, indent=4)

    print(f"Created dummy config files in {dummy_configs_dir}")

    # Test loading YAML
    try:
        print("\n--- Testing YAML load ---")
        config_yaml = load_config(config_path=dummy_yaml_path)
        print("YAML Config loaded successfully:")
        print(f"Project: {config_yaml.project_name}")
    except ZeffyConfigurationError as e:
        print(f"Error loading YAML: {e}")

    # Test loading JSON
    try:
        print("\n--- Testing JSON load ---")
        config_json = load_config(config_path=dummy_json_path)
        print("JSON Config loaded successfully:")
        print(f"Project: {config_json.project_name}")
    except ZeffyConfigurationError as e:
        print(f"Error loading JSON: {e}")

    # Test loading with a dictionary
    try:
        print("\n--- Testing Dict load ---")
        config_dict_loaded = load_config(config_dict=sample_data_for_loader)
        print("Dict Config loaded successfully:")
        print(f"Project: {config_dict_loaded.project_name}")
    except ZeffyConfigurationError as e:
        print(f"Error loading from dict: {e}")

    # Test with missing required field (e.g., models)
    invalid_data = {
        "project_name": "Invalid Project",
        "data_loader": {"type": "csv", "path": "/data/input.csv", "target_column": "outcome"}
    }
    try:
        print("\n--- Testing Invalid Dict load (missing models) ---")
        load_config(config_dict=invalid_data)
    except ZeffyConfigurationError as e:
        print(f"Correctly caught ZeffyConfigurationError for missing models: {e}")

    # Test with no config (should raise ZeffyConfigurationError due to missing required fields)
    try:
        print("\n--- Testing No Config load ---")
        load_config()
    except ZeffyConfigurationError as e:
        print(f"Correctly caught ZeffyConfigurationError for no config: {e}")
        
    # Test non-existent file
    try:
        print("\n--- Testing Non-existent file load ---")
        load_config(config_path="/home/ubuntu/non_existent_config.yaml")
    except ZeffyConfigurationError as e:
        print(f"Correctly caught ZeffyConfigurationError for non-existent file: {e}")

    # Test unsupported file type
    unsupported_file_path = os.path.join(dummy_configs_dir, "test_config.txt")
    with open(unsupported_file_path, 'w') as f:
        f.write("this is not a yaml or json")
    try:
        print("\n--- Testing Unsupported file type load ---")
        load_config(config_path=unsupported_file_path)
    except ZeffyConfigurationError as e:
        print(f"Correctly caught ZeffyConfigurationError for unsupported file type: {e}")

    print("\nConfig loader tests finished.")


# --- Zeffy Base Model --- 
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for all models in Zeffy."""
    def __init__(self, model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.task_type = task_type # "classification" or "regression"
        self.params = params if params is not None else {}
        self.model: Any = None # This will hold the actual trained model instance
        self.is_trained: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, **kwargs) -> None:
        """Trains the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        """Generates predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Optional[Any]:
        """Generates probability predictions (for classification tasks)."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Returns the model parameters."""
        return self.params

    def set_params(self, **params) -> None:
        """Sets the model parameters."""
        self.params.update(params)
        # Re-initialize model if necessary, or handle during fit

    def save_model(self, path: str) -> None:
        """Saves the trained model to a file."""
        # Generic save, specific models might override or use a helper
        # For example, using joblib or dill
        import joblib # Or dill, or model-specific save methods
        try:
            joblib.dump(self.model, path)
            print(f"Model {self.model_type} saved to {path}")
        except Exception as e:
            print(f"Error saving model {self.model_type} to {path}: {e}")
            raise

    def load_model(self, path: str) -> None:
        """Loads a trained model from a file."""
        import joblib
        try:
            self.model = joblib.load(path)
            self.is_trained = True
            print(f"Model {self.model_type} loaded from {path}")
        except Exception as e:
            print(f"Error loading model {self.model_type} from {path}: {e}")
            raise

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Returns feature importances if the model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else range(len(self.model.feature_importances_)),
                'importance': self.model.feature_importances_
            }).sort_values(by='importance', ascending=False)
        elif hasattr(self.model, 'coef_'): # For linear models
             return pd.DataFrame({
                'feature': self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else range(len(self.model.coef_)),
                'importance': self.model.coef_.flatten() # Flatten in case of multi-class coefs
            }).sort_values(by='importance', key=abs, ascending=False)
        print(f"Feature importance not directly available for model type {self.model_type}")
        return None

    def __str__(self):
        return f"{self.__class__.__name__}(model_type=\'{self.model_type}\', task_type=\'{self.task_type}\', params={self.params})"


# --- Zeffy Model Registry --- 
from typing import Dict, Type, Callable
# from .base_model import BaseModel # Zeffy: Relative import handled by consolidation order

class ModelRegistry:
    """A registry for model classes and their instantiation."""
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """Decorator to register a model class."""
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            if model_type in cls._registry:
                print(f"Warning: Model type 	hemed_text_color_red_bold_start_bold_end_themed_text_color_end{model_type}	hemed_text_color_red_bold_start_bold_end_themed_text_color_end is already registered. Overwriting.")
            cls._registry[model_type] = model_class
            print(f"Model 	hemed_text_color_green_bold_start_bold_end_themed_text_color_end{model_type}	hemed_text_color_green_bold_start_bold_end_themed_text_color_end registered with class 	hemed_text_color_green_bold_start_bold_end_themed_text_color_end{model_class.__name__}	hemed_text_color_green_bold_start_bold_end_themed_text_color_end")
            return model_class
        return decorator

    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseModel]:
        """Retrieves a model class from the registry."""
        if model_type not in cls._registry:
            raise ValueError(f"Model type 	hemed_text_color_red_bold_start_bold_end_themed_text_color_end{model_type}	hemed_text_color_red_bold_start_bold_end_themed_text_color_end not found in registry. Available models: {list(cls._registry.keys())}")
        return cls._registry[model_type]

    @classmethod
    def create_model(cls, model_type: str, task_type: str, params: Dict = None) -> BaseModel:
        """Creates an instance of a registered model."""
        model_class = cls.get_model_class(model_type)
        try:
            return model_class(model_type=model_type, task_type=task_type, params=params or {})
        except Exception as e:
            print(f"Error instantiating model 	hemed_text_color_red_bold_start_bold_end_themed_text_color_end{model_type}	hemed_text_color_red_bold_start_bold_end_themed_text_color_end of class 	hemed_text_color_red_bold_start_bold_end_themed_text_color_end{model_class.__name__}	hemed_text_color_red_bold_start_bold_end_themed_text_color_end: {e}")
            raise

# Convenience functions for direct use
def register_model(model_type: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    return ModelRegistry.register(model_type)

def get_model(model_type: str, task_type: str, params: Dict = None) -> BaseModel:
    return ModelRegistry.create_model(model_type, task_type, params)

# if __name__ == "__main__": # Zeffy: Main block from module, commented out.
    # Example Usage of ModelRegistry

    @register_model("dummy_classifier")
    class DummyClassifier(BaseModel):
        def fit(self, X, y, X_val=None, y_val=None, **kwargs):
            print(f"Fitting DummyClassifier with params: {self.params} on data shape {X.shape}")
            self.model = "trained_dummy_classifier"
            self.is_trained = True

        def predict(self, X, **kwargs):
            if not self.is_trained:
                raise RuntimeError("Model not trained yet.")
            print(f"Predicting with DummyClassifier on data shape {X.shape}")
            return [0] * len(X) # Dummy predictions

        def predict_proba(self, X, **kwargs):
            if not self.is_trained:
                raise RuntimeError("Model not trained yet.")
            print(f"Predicting probabilities with DummyClassifier on data shape {X.shape}")
            return [[1.0, 0.0]] * len(X) # Dummy probabilities

    @register_model("dummy_regressor")
    class DummyRegressor(BaseModel):
        def fit(self, X, y, X_val=None, y_val=None, **kwargs):
            print(f"Fitting DummyRegressor with params: {self.params} on data shape {X.shape}")
            self.model = "trained_dummy_regressor"
            self.is_trained = True

        def predict(self, X, **kwargs):
            if not self.is_trained:
                raise RuntimeError("Model not trained yet.")
            print(f"Predicting with DummyRegressor on data shape {X.shape}")
            return [0.0] * len(X) # Dummy predictions

        def predict_proba(self, X, **kwargs):
            # Regression models typically don't have predict_proba
            return None

    # Test creating models
    try:
        print("\n--- Testing Model Creation ---")
        classifier_params = {"C": 1.0, "solver": "liblinear"}
        dummy_clf = get_model("dummy_classifier", task_type="classification", params=classifier_params)
        print(f"Created model: {dummy_clf}")
        # dummy_clf.fit(pd.DataFrame([[1,2],[3,4]]), pd.Series([0,1])) # Requires pandas
        # dummy_clf.predict(pd.DataFrame([[5,6]])) # Requires pandas

        regressor_params = {"alpha": 0.5}
        dummy_reg = get_model("dummy_regressor", task_type="regression", params=regressor_params)
        print(f"Created model: {dummy_reg}")

        # Test getting a non-existent model
        print("\n--- Testing Non-existent Model ---")
        get_model("non_existent_model", task_type="classification")

    except ValueError as e:
        print(f"Correctly caught error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\nModel registry tests finished.")


# --- Zeffy Model Implementations --- 
import lightgbm as lgb
import pandas as pd
from typing import Any, Dict, Optional

# from ..base_model import BaseModel # Zeffy: Relative import handled by consolidation order
# from ..registry import register_model # Zeffy: Relative import handled by consolidation order

@register_model("lightgbm_classifier")
class LightGBMClassifier(BaseModel):
    """LightGBM Classifier model."""

    def __init__(self, model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, task_type, params)
        if self.task_type != "classification":
            raise ValueError(f"LightGBMClassifier is for classification tasks, but task_type is {self.task_type}")
        self._default_params = {
            "objective": "binary", # or "multiclass"
            "metric": "auc", # or "multi_logloss"
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "random_state": self.params.get("random_state", 42), # Use from global config if possible
            "n_jobs": -1,
            "verbose": -1,
        }
        self.params = {**self._default_params, **self.params} # Merge user params with defaults

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, **kwargs) -> None:
        """Trains the LightGBM classifier."""
        if self.params.get("objective") == "multiclass" and "num_class" not in self.params:
            self.params["num_class"] = y.nunique()
            print(f"Inferred num_class: {self.params['num_class']} for multiclass LightGBM.")

        self.model = lgb.LGBMClassifier(**self.params)
        
        eval_set = []
        callbacks = []

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # Add early stopping callback if configured
            early_stopping_rounds = kwargs.get("early_stopping_rounds", self.params.get("early_stopping_rounds", None))
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=-1))
        
        print(f"Fitting LightGBMClassifier with params: {self.model.get_params()}")
        try:
            self.model.fit(
                X, y, 
                eval_set=eval_set if eval_set else None,
                callbacks=callbacks if callbacks else None
            )
            self.is_trained = True
            print("LightGBMClassifier training completed.")
        except Exception as e:
            print(f"Error during LightGBMClassifier training: {e}")
            # Potentially log the full traceback
            raise

    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        """Generates predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        try:
            return self.model.predict(X, **kwargs)
        except Exception as e:
            print(f"Error during LightGBMClassifier prediction: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Optional[Any]:
        """Generates probability predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        if not hasattr(self.model, "predict_proba"):
            print(f"Model {self.model_type} does not support predict_proba.")
            return None
        try:
            return self.model.predict_proba(X, **kwargs)
        except Exception as e:
            print(f"Error during LightGBMClassifier probability prediction: {e}")
            raise

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Returns feature importances if the model supports it."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            print("Model not trained or feature importances not available.")
            return None
        
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        if hasattr(self.model, 'feature_name_') and list(self.model.feature_name_) != ['Column_0', 'Column_1'] : # default names if not set
             feature_names = self.model.feature_name_
        elif hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_

        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)
        return importance_df

# Example of how to test this specific model (can be run with python -m zeffy.models.classification.lightgbm_classifier)
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np

    print("Testing LightGBMClassifier...")
    X_np, y_np = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
    y_s = pd.Series(y_np)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2

    # Binary classification test
    print("\n--- Binary Classification Test ---")
    lgbm_clf_binary_params = {
        "n_estimators": 50,
        "learning_rate": 0.05,
        "random_state": 42,
        "objective": "binary",
        "metric": "binary_logloss,auc"
    }
    lgbm_binary_model = LightGBMClassifier(model_type="lightgbm_classifier", task_type="classification", params=lgbm_clf_binary_params)
    print(f"Initial params: {lgbm_binary_model.params}")
    lgbm_binary_model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=10)
    
    binary_preds = lgbm_binary_model.predict(X_test)
    print(f"Binary predictions (first 5): {binary_preds[:5]}")
    binary_probas = lgbm_binary_model.predict_proba(X_test)
    print(f"Binary probabilities (first 5):\n{binary_probas[:5]}")
    
    feature_imp = lgbm_binary_model.get_feature_importance()
    if feature_imp is not None:
        print(f"Feature importances (top 5):\n{feature_imp.head()}")

    # Multiclass classification test
    print("\n--- Multiclass Classification Test ---")
    X_multi_np, y_multi_np = make_classification(n_samples=300, n_features=20, n_informative=15, n_classes=3, random_state=42)
    X_multi_df = pd.DataFrame(X_multi_np, columns=[f'feature_m_{i}' for i in range(X_multi_np.shape[1])])
    y_multi_s = pd.Series(y_multi_np)
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_multi_df, y_multi_s, test_size=0.2, random_state=42)
    X_m_train, X_m_val, y_m_train, y_m_val = train_test_split(X_m_train, y_m_train, test_size=0.25, random_state=42)

    lgbm_clf_multi_params = {
        "n_estimators": 60,
        "learning_rate": 0.08,
        "random_state": 123,
        "objective": "multiclass", # num_class will be inferred
        "metric": "multi_logloss"
    }
    lgbm_multi_model = LightGBMClassifier(model_type="lightgbm_classifier", task_type="classification", params=lgbm_clf_multi_params)
    lgbm_multi_model.fit(X_m_train, y_m_train, X_val=X_m_val, y_m_val=y_m_val, early_stopping_rounds=5)

    multi_preds = lgbm_multi_model.predict(X_m_test)
    print(f"Multiclass predictions (first 5): {multi_preds[:5]}")
    multi_probas = lgbm_multi_model.predict_proba(X_m_test)
    print(f"Multiclass probabilities (first 5):\n{multi_probas[:5]}")

    print("\nLightGBMClassifier tests finished.")


import lightgbm as lgb
import pandas as pd
from typing import Any, Dict, Optional

# from ..base_model import BaseModel # Zeffy: Relative import handled by consolidation order
# from ..registry import register_model # Zeffy: Relative import handled by consolidation order

@register_model("lightgbm_regressor")
class LightGBMRegressor(BaseModel):
    """LightGBM Regressor model."""

    def __init__(self, model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, task_type, params)
        if self.task_type != "regression":
            raise ValueError(f"LightGBMRegressor is for regression tasks, but task_type is {self.task_type}")
        self._default_params = {
            "objective": "regression", # Common objectives: regression_l1, regression_l2, huber, poisson, etc.
            "metric": "rmse", # Common metrics: rmse, mae, rmsle
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "random_state": self.params.get("random_state", 42),
            "n_jobs": -1,
            "verbose": -1,
        }
        self.params = {**self._default_params, **self.params} # Merge user params with defaults

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, **kwargs) -> None:
        """Trains the LightGBM regressor."""
        self.model = lgb.LGBMRegressor(**self.params)
        
        eval_set = []
        callbacks = []

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            early_stopping_rounds = kwargs.get("early_stopping_rounds", self.params.get("early_stopping_rounds", None))
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=-1))
        
        print(f"Fitting LightGBMRegressor with params: {self.model.get_params()}")
        try:
            self.model.fit(
                X, y, 
                eval_set=eval_set if eval_set else None,
                callbacks=callbacks if callbacks else None
            )
            self.is_trained = True
            print("LightGBMRegressor training completed.")
        except Exception as e:
            print(f"Error during LightGBMRegressor training: {e}")
            raise

    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        """Generates predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        try:
            return self.model.predict(X, **kwargs)
        except Exception as e:
            print(f"Error during LightGBMRegressor prediction: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Optional[Any]:
        """Regressor does not support predict_proba."""
        print(f"Model {self.model_type} (Regressor) does not support predict_proba.")
        return None

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Returns feature importances if the model supports it."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            print("Model not trained or feature importances not available.")
            return None
        
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        if hasattr(self.model, 'feature_name_') and list(self.model.feature_name_) != ['Column_0', 'Column_1']:
             feature_names = self.model.feature_name_
        elif hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_

        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)
        return importance_df

# Example of how to test this specific model
if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np

    print("Testing LightGBMRegressor...")
    X_np, y_np = make_regression(n_samples=200, n_features=20, n_informative=10, random_state=42)
    X_df = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
    y_s = pd.Series(y_np)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    lgbm_reg_params = {
        "n_estimators": 75,
        "learning_rate": 0.07,
        "random_state": 123,
        "objective": "regression_l2",
        "metric": "rmse"
    }
    lgbm_reg_model = LightGBMRegressor(model_type="lightgbm_regressor", task_type="regression", params=lgbm_reg_params)
    print(f"Initial params: {lgbm_reg_model.params}")
    lgbm_reg_model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=10)
    
    reg_preds = lgbm_reg_model.predict(X_test)
    print(f"Regression predictions (first 5): {reg_preds[:5]}")
    
    rmse = mean_squared_error(y_test, reg_preds, squared=False)
    print(f"RMSE on test set: {rmse:.4f}")

    feature_imp = lgbm_reg_model.get_feature_importance()
    if feature_imp is not None:
        print(f"Feature importances (top 5):\n{feature_imp.head()}")

    # Test predict_proba (should indicate not supported)
    lgbm_reg_model.predict_proba(X_test)

    print("\nLightGBMRegressor tests finished.")


import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from typing import Any, Dict, Optional

# from ..base_model import BaseModel # Zeffy: Relative import handled by consolidation order
# from ..registry import register_model # Zeffy: Relative import handled by consolidation order

@register_model("sklearn_logistic_regression")
class SklearnLogisticRegressionModel(BaseModel):
    """Scikit-learn Logistic Regression model for classification."""

    def __init__(self, model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, task_type, params)
        if self.task_type != "classification":
            raise ValueError(f"SklearnLogisticRegressionModel is for classification tasks, but task_type is {self.task_type}")
        
        self._default_params = {
            "solver": "liblinear", # Good default for many cases
            "random_state": self.params.get("random_state", 42),
            "C": 1.0,
            "penalty": "l2",
            "max_iter": 100,
            "n_jobs": -1
        }
        self.params = {**self._default_params, **self.params}

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, **kwargs) -> None:
        """Trains the Scikit-learn Logistic Regression model."""
        # Sklearn models generally don't use X_val, y_val directly in fit for early stopping like GBDTs
        # However, they could be used for custom callbacks or logging if needed.
        
        # Ensure params are compatible with SklearnLogisticRegression
        current_params = self.params.copy()
        # n_jobs is not a direct param for LogisticRegression constructor but can be used by some solvers internally or for CV
        # For simplicity, we remove it if not directly supported or handle it if a CV search is integrated here.
        if "n_jobs" in current_params and current_params.get("solver") not in ["saga"] : # saga solver uses n_jobs
            #del current_params["n_jobs"] # or handle appropriately
            pass # keep it for now, sklearn might ignore if not applicable

        self.model = SklearnLogisticRegression(**current_params)
        
        print(f"Fitting SklearnLogisticRegressionModel with params: {self.model.get_params()}")
        try:
            self.model.fit(X, y)
            self.is_trained = True
            print("SklearnLogisticRegressionModel training completed.")
        except Exception as e:
            print(f"Error during SklearnLogisticRegressionModel training: {e}")
            raise

    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        """Generates predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        try:
            return self.model.predict(X, **kwargs)
        except Exception as e:
            print(f"Error during SklearnLogisticRegressionModel prediction: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Optional[Any]:
        """Generates probability predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        if not hasattr(self.model, "predict_proba"):
            print(f"Model {self.model_type} does not support predict_proba.") # Should not happen for LogisticRegression
            return None
        try:
            return self.model.predict_proba(X, **kwargs)
        except Exception as e:
            print(f"Error during SklearnLogisticRegressionModel probability prediction: {e}")
            raise

# Example of how to test this specific model
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("Testing SklearnLogisticRegressionModel...")
    X_np, y_np = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
    y_s = pd.Series(y_np)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.25, random_state=42)

    # Binary classification test
    print("\n--- Binary Classification Test ---")
    logreg_params = {
        "C": 0.5,
        "random_state": 123,
        "solver": "liblinear",
        "max_iter": 200
    }
    logreg_model = SklearnLogisticRegressionModel(model_type="sklearn_logistic_regression", task_type="classification", params=logreg_params)
    print(f"Initial params: {logreg_model.params}")
    logreg_model.fit(X_train, y_train)
    
    binary_preds = logreg_model.predict(X_test)
    binary_probas = logreg_model.predict_proba(X_test)

    acc = accuracy_score(y_test, binary_preds)
    auc = roc_auc_score(y_test, binary_probas[:, 1])
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Binary predictions (first 5): {binary_preds[:5]}")
    print(f"Binary probabilities (first 5):\n{binary_probas[:5]}")
    
    feature_imp = logreg_model.get_feature_importance()
    if feature_imp is not None:
        print(f"Feature importances (coefficients) (top 5):\n{feature_imp.head()}")

    # Test with multiclass (LogisticRegression handles it automatically)
    print("\n--- Multiclass Classification Test (using LogisticRegression OVR) ---")
    X_multi_np, y_multi_np = make_classification(n_samples=300, n_features=20, n_informative=15, n_classes=3, random_state=42)
    X_multi_df = pd.DataFrame(X_multi_np, columns=[f'feature_m_{i}' for i in range(X_multi_np.shape[1])])
    y_multi_s = pd.Series(y_multi_np)
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_multi_df, y_multi_s, test_size=0.25, random_state=42)

    logreg_multi_params = {"random_state": 42, "solver": "lbfgs", "multi_class": "ovr"} # lbfgs supports multiclass
    logreg_multi_model = SklearnLogisticRegressionModel(model_type="sklearn_logistic_regression", task_type="classification", params=logreg_multi_params)
    logreg_multi_model.fit(X_m_train, y_m_train)

    multi_preds = logreg_multi_model.predict(X_m_test)
    multi_probas = logreg_multi_model.predict_proba(X_m_test)
    multi_acc = accuracy_score(y_m_test, multi_preds)
    # roc_auc_score needs one-hot encoded y_true for multiclass or use ovo/ovr options
    print(f"Multiclass Test Accuracy: {multi_acc:.4f}")
    print(f"Multiclass predictions (first 5): {multi_preds[:5]}")
    print(f"Multiclass probabilities (first 5):\n{multi_probas[:5]}")

    print("\nSklearnLogisticRegressionModel tests finished.")


import xgboost as xgb
import pandas as pd
from typing import Any, Dict, Optional

# from ..base_model import BaseModel # Zeffy: Relative import handled by consolidation order
# from ..registry import register_model # Zeffy: Relative import handled by consolidation order

@register_model("xgboost_classifier")
class XGBoostClassifier(BaseModel):
    """XGBoost Classifier model."""

    def __init__(self, model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, task_type, params)
        if self.task_type != "classification":
            raise ValueError(f"XGBoostClassifier is for classification tasks, but task_type is {self.task_type}")
        
        self._default_params = {
            "objective": "binary:logistic", # or "multi:softprob"
            "eval_metric": "logloss", # or "mlogloss"
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": self.params.get("random_state", 42),
            "n_jobs": -1,
            "use_label_encoder": False, # Suppress warning for newer XGBoost versions
        }
        self.params = {**self._default_params, **self.params} # Merge user params with defaults

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, **kwargs) -> None:
        """Trains the XGBoost classifier."""
        if self.params.get("objective") == "multi:softprob" and "num_class" not in self.params:
            self.params["num_class"] = y.nunique()
            print('Inferred num_class: {} for multiclass XGBoost.'.format(self.params["num_class"]))

        # Ensure use_label_encoder is set appropriately based on XGBoost version if not explicitly passed
        # For XGBoost >= 1.6.0, label encoding is handled internally or not needed for numeric labels.
        # The `use_label_encoder=False` in defaults is generally good for newer versions.

        self.model = xgb.XGBClassifier(**self.params)
        
        eval_set_list = []
        fit_params = {}

        if X_val is not None and y_val is not None:
            eval_set_list = [(X_val, y_val)]
            early_stopping_rounds = kwargs.get("early_stopping_rounds", self.params.get("early_stopping_rounds", None))
            if early_stopping_rounds:
                fit_params["early_stopping_rounds"] = early_stopping_rounds
                fit_params["verbose"] = False # To suppress messages from early stopping unless verbose is explicitly set in params
        
        print(f"Fitting XGBoostClassifier with params: {self.model.get_params()}")
        try:
            self.model.fit(
                X, y, 
                eval_set=eval_set_list if eval_set_list else None,
                **fit_params
            )
            self.is_trained = True
            print("XGBoostClassifier training completed.")
        except Exception as e:
            print(f"Error during XGBoostClassifier training: {e}")
            raise

    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        """Generates predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        try:
            return self.model.predict(X, **kwargs)
        except Exception as e:
            print(f"Error during XGBoostClassifier prediction: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Optional[Any]:
        """Generates probability predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        if not hasattr(self.model, "predict_proba"):
            print(f"Model {self.model_type} does not support predict_proba.")
            return None
        try:
            return self.model.predict_proba(X, **kwargs)
        except Exception as e:
            print(f"Error during XGBoostClassifier probability prediction: {e}")
            raise

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            print("Model not trained or feature importances not available for XGBoost.")
            return None

        try:
            feature_names = self.model.get_booster().feature_names
            if feature_names is None and hasattr(X, 'columns'): # Fallback if booster names not set
                feature_names = X.columns.tolist()
            elif feature_names is None:
                 feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]

            importances = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False).reset_index(drop=True)
            return importance_df
        except Exception as e:
            print(f"Could not retrieve feature importance for XGBoost: {e}")
            return None

# Example of how to test this specific model
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np

    print("Testing XGBoostClassifier...")
    X_np, y_np = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
    y_s = pd.Series(y_np)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42)
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Binary classification test
    print("\n--- Binary Classification Test ---")
    xgb_clf_binary_params = {
        "n_estimators": 50,
        "learning_rate": 0.05,
        "random_state": 42,
        "objective": "binary:logistic",
        "eval_metric": "logloss"
    }
    xgb_binary_model = XGBoostClassifier(model_type="xgboost_classifier", task_type="classification", params=xgb_clf_binary_params)
    print(f"Initial params: {xgb_binary_model.params}")
    xgb_binary_model.fit(X_train_xgb, y_train_xgb, X_val=X_val_xgb, y_val=y_val_xgb, early_stopping_rounds=10)
    
    binary_preds = xgb_binary_model.predict(X_test)
    print(f"Binary predictions (first 5): {binary_preds[:5]}")
    binary_probas = xgb_binary_model.predict_proba(X_test)
    print(f"Binary probabilities (first 5):\n{binary_probas[:5]}")
    
    feature_imp = xgb_binary_model.get_feature_importance()
    if feature_imp is not None:
        print(f"Feature importances (top 5):\n{feature_imp.head()}")

    # Multiclass classification test
    print("\n--- Multiclass Classification Test ---")
    X_multi_np, y_multi_np = make_classification(n_samples=300, n_features=20, n_informative=15, n_classes=3, random_state=42)
    X_multi_df = pd.DataFrame(X_multi_np, columns=[f'feature_m_{i}' for i in range(X_multi_np.shape[1])])
    y_multi_s = pd.Series(y_multi_np)
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_multi_df, y_multi_s, test_size=0.2, random_state=42)
    X_m_train_xgb, X_m_val_xgb, y_m_train_xgb, y_m_val_xgb = train_test_split(X_m_train, y_m_train, test_size=0.25, random_state=42)

    xgb_clf_multi_params = {
        "n_estimators": 60,
        "learning_rate": 0.08,
        "random_state": 123,
        "objective": "multi:softprob", # num_class will be inferred
        "eval_metric": "mlogloss"
    }
    xgb_multi_model = XGBoostClassifier(model_type="xgboost_classifier", task_type="classification", params=xgb_clf_multi_params)
    xgb_multi_model.fit(X_m_train_xgb, y_m_train_xgb, X_val=X_m_val_xgb, y_val=y_m_val_xgb, early_stopping_rounds=5)

    multi_preds = xgb_multi_model.predict(X_m_test)
    print(f"Multiclass predictions (first 5): {multi_preds[:5]}")
    multi_probas = xgb_multi_model.predict_proba(X_m_test)
    print(f"Multiclass probabilities (first 5):\n{multi_probas[:5]}")

    print("\nXGBoostClassifier tests finished.")


import xgboost as xgb
import pandas as pd
from typing import Any, Dict, Optional

# from ..base_model import BaseModel # Zeffy: Relative import handled by consolidation order
# from ..registry import register_model # Zeffy: Relative import handled by consolidation order

@register_model("xgboost_regressor")
class XGBoostRegressor(BaseModel):
    """XGBoost Regressor model."""

    def __init__(self, model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, task_type, params)
        if self.task_type != "regression":
            raise ValueError(f"XGBoostRegressor is for regression tasks, but task_type is {self.task_type}")
        
        self._default_params = {
            "objective": "reg:squarederror", # Common objectives: reg:squarederror, reg:linear, reg:logistic, count:poisson
            "eval_metric": "rmse", # Common metrics: rmse, mae
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": self.params.get("random_state", 42),
            "n_jobs": -1,
        }
        self.params = {**self._default_params, **self.params} # Merge user params with defaults

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, **kwargs) -> None:
        """Trains the XGBoost regressor."""
        self.model = xgb.XGBRegressor(**self.params)
        
        eval_set_list = []
        fit_params = {}

        if X_val is not None and y_val is not None:
            eval_set_list = [(X_val, y_val)]
            early_stopping_rounds = kwargs.get("early_stopping_rounds", self.params.get("early_stopping_rounds", None))
            if early_stopping_rounds:
                fit_params["early_stopping_rounds"] = early_stopping_rounds
                fit_params["verbose"] = False
        
        print(f"Fitting XGBoostRegressor with params: {self.model.get_params()}")
        try:
            self.model.fit(
                X, y, 
                eval_set=eval_set_list if eval_set_list else None,
                **fit_params
            )
            self.is_trained = True
            print("XGBoostRegressor training completed.")
        except Exception as e:
            print(f"Error during XGBoostRegressor training: {e}")
            raise

    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        """Generates predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        try:
            return self.model.predict(X, **kwargs)
        except Exception as e:
            print(f"Error during XGBoostRegressor prediction: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Optional[Any]:
        """Regressor does not support predict_proba."""
        print(f"Model {self.model_type} (Regressor) does not support predict_proba.")
        return None

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            print("Model not trained or feature importances not available for XGBoost Regressor.")
            return None
        try:
            feature_names = self.model.get_booster().feature_names
            if feature_names is None and hasattr(X, 'columns'): # Fallback if booster names not set
                feature_names = X.columns.tolist()
            elif feature_names is None:
                 feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]

            importances = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False).reset_index(drop=True)
            return importance_df
        except Exception as e:
            print(f"Could not retrieve feature importance for XGBoost Regressor: {e}")
            return None

# Example of how to test this specific model
if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np

    print("Testing XGBoostRegressor...")
    X_np, y_np = make_regression(n_samples=200, n_features=20, n_informative=10, random_state=42)
    X_df = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
    y_s = pd.Series(y_np)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42)
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    xgb_reg_params = {
        "n_estimators": 75,
        "learning_rate": 0.07,
        "random_state": 123,
        "objective": "reg:squarederror",
        "eval_metric": "rmse"
    }
    xgb_reg_model = XGBoostRegressor(model_type="xgboost_regressor", task_type="regression", params=xgb_reg_params)
    print(f"Initial params: {xgb_reg_model.params}")
    xgb_reg_model.fit(X_train_xgb, y_train_xgb, X_val=X_val_xgb, y_val=y_val_xgb, early_stopping_rounds=10)
    
    reg_preds = xgb_reg_model.predict(X_test)
    print(f"Regression predictions (first 5): {reg_preds[:5]}")
    
    rmse = mean_squared_error(y_test, reg_preds, squared=False)
    print(f"RMSE on test set: {rmse:.4f}")

    feature_imp = xgb_reg_model.get_feature_importance()
    if feature_imp is not None:
        print(f"Feature importances (top 5):\n{feature_imp.head()}")

    # Test predict_proba (should indicate not supported)
    xgb_reg_model.predict_proba(X_test)

    print("\nXGBoostRegressor tests finished.")


from catboost import CatBoostClassifier as CBCModel
import pandas as pd
from typing import Any, Dict, Optional, List

# from ..base_model import BaseModel # Zeffy: Relative import handled by consolidation order
# from ..registry import register_model # Zeffy: Relative import handled by consolidation order

@register_model("catboost_classifier")
class CatBoostClassifier(BaseModel):
    """CatBoost Classifier model."""

    def __init__(self, model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, task_type, params)
        if self.task_type != "classification":
            raise ValueError(f"CatBoostClassifier is for classification tasks, but task_type is {self.task_type}")
        
        self._default_params = {
            "objective": "Logloss", # or "MultiClass"
            "eval_metric": "AUC", # or "MultiClass"
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 6,
            "random_seed": self.params.get("random_state", self.params.get("random_seed", 42)), # CatBoost uses random_seed
            "verbose": 0, # Suppress CatBoost verbosity by default
            "early_stopping_rounds": None # Handled in fit method
        }
        # User params override defaults. If random_state is in user_params, it overrides random_seed from defaults.
        # CatBoost uses `random_seed` not `random_state`.
        user_params = params if params is not None else {}
        if "random_state" in user_params and "random_seed" not in user_params:
            user_params["random_seed"] = user_params.pop("random_state")
        
        self.params = {**self._default_params, **user_params}
        if self.params["verbose"] is None or self.params["verbose"] < 100 and self.params["verbose"] !=0 : # make it less verbose unless user specified higher
            self.params["logging_level"] = "Silent"
        else:
            self.params["logging_level"] = "Verbose"


    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, cat_features: Optional[List[int]] = None, **kwargs) -> None:
        """Trains the CatBoost classifier."""
        
        fit_params = self.params.copy()
        # CatBoost specific handling for early stopping
        early_stopping_rounds_val = kwargs.get("early_stopping_rounds", fit_params.pop("early_stopping_rounds", None))

        if fit_params.get("objective") == "MultiClass" and "classes_count" not in fit_params:
            fit_params["classes_count"] = y.nunique()
            print(f"Inferred classes_count: {fit_params['classes_count']} for multiclass CatBoost.")

        self.model = CBCModel(**fit_params)
        
        eval_set_list = None
        if X_val is not None and y_val is not None:
            eval_set_list = (X_val, y_val)
        
        # Identify categorical features if not provided (can be slow for many columns)
        if cat_features is None:
            cat_features = [i for i, col_type in enumerate(X.dtypes) if str(col_type) == "category" or X.iloc[:, i].nunique() < self.params.get("one_hot_max", 20) and X.iloc[:,i].dtype==object]
            if cat_features:
                print(f"Inferred categorical features indices for CatBoost: {cat_features}")

        print(f"Fitting CatBoostClassifier with effective params: {self.model.get_params()}")
        try:
            self.model.fit(
                X, y, 
                eval_set=eval_set_list,
                cat_features=cat_features if cat_features else None,
                early_stopping_rounds=early_stopping_rounds_val,
                # verbose is controlled by logging_level in params for CatBoost
            )
            self.is_trained = True
            print("CatBoostClassifier training completed.")
        except Exception as e:
            print(f"Error during CatBoostClassifier training: {e}")
            raise

    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        try:
            return self.model.predict(X, **kwargs)
        except Exception as e:
            print(f"Error during CatBoostClassifier prediction: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Optional[Any]:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        if not hasattr(self.model, "predict_proba"):
            print(f"Model {self.model_type} does not support predict_proba.")
            return None
        try:
            return self.model.predict_proba(X, **kwargs)
        except Exception as e:
            print(f"Error during CatBoostClassifier probability prediction: {e}")
            raise

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_trained or not hasattr(self.model, "get_feature_importance"):
            print("Model not trained or feature importances not available for CatBoost.")
            return None
        try:
            feature_names = self.model.feature_names_ if self.model.feature_names_ else [f"feature_{i}" for i in range(len(self.model.get_feature_importance()))]
            importances = self.model.get_feature_importance()
            
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
            return importance_df
        except Exception as e:
            print(f"Could not retrieve feature importance for CatBoost: {e}")
            return None

# if __name__ == "__main__": # Zeffy: Main block from module, commented out.
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("Testing CatBoostClassifier...")
    X_np, y_np = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X_np, columns=[f"feature_{i}" for i in range(X_np.shape[1])])
    # Make some features categorical for CatBoost to handle
    X_df["feature_cat_1"] = pd.Series(np.random.choice(["A", "B", "C"], size=len(X_df))).astype("category")
    X_df["feature_cat_2"] = pd.Series(np.random.choice(["X", "Y"], size=len(X_df))).astype("category")
    y_s = pd.Series(y_np)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42)
    X_train_cb, X_val_cb, y_train_cb, y_val_cb = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print("\n--- Binary Classification Test ---")
    cb_clf_binary_params = {
        "iterations": 50,
        "learning_rate": 0.05,
        "random_seed": 42,
        "objective": "Logloss",
        "eval_metric": "Logloss,AUC",
        "verbose": 0 # Keep it silent for tests
    }
    cb_binary_model = CatBoostClassifier(model_type="catboost_classifier", task_type="classification", params=cb_clf_binary_params)
    cb_binary_model.fit(X_train_cb, y_train_cb, X_val=X_val_cb, y_val=y_val_cb, early_stopping_rounds=10)
    
    binary_preds = cb_binary_model.predict(X_test)
    print(f"Binary predictions (first 5): {binary_preds[:5].flatten().tolist()}")
    binary_probas = cb_binary_model.predict_proba(X_test)
    print(f"Binary probabilities (first 5):\n{binary_probas[:5]}")
    
    feature_imp = cb_binary_model.get_feature_importance()
    if feature_imp is not None:
        print(f"Feature importances (top 5):\n{feature_imp.head()}")

    print("\n--- Multiclass Classification Test ---")
    X_multi_np, y_multi_np = make_classification(n_samples=300, n_features=20, n_informative=15, n_classes=3, random_state=42)
    X_multi_df = pd.DataFrame(X_multi_np, columns=[f"feature_m_{i}" for i in range(X_multi_np.shape[1])])
    X_multi_df["feature_m_cat_1"] = pd.Series(np.random.choice(["P", "Q", "R", "S"], size=len(X_multi_df))).astype("category")
    y_multi_s = pd.Series(y_multi_np)
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_multi_df, y_multi_s, test_size=0.2, random_state=42)
    X_m_train_cb, X_m_val_cb, y_m_train_cb, y_m_val_cb = train_test_split(X_m_train, y_m_train, test_size=0.25, random_state=42)

    cb_clf_multi_params = {
        "iterations": 60,
        "learning_rate": 0.08,
        "random_seed": 123,
        "objective": "MultiClass",
        "eval_metric": "MultiClass",
        "verbose": 0
    }
    cb_multi_model = CatBoostClassifier(model_type="catboost_classifier", task_type="classification", params=cb_clf_multi_params)
    cb_multi_model.fit(X_m_train_cb, y_m_train_cb, X_val=X_m_val_cb, y_val=y_m_val_cb, early_stopping_rounds=5)

    multi_preds = cb_multi_model.predict(X_m_test)
    print(f"Multiclass predictions (first 5): {multi_preds[:5].flatten().tolist()}")
    multi_probas = cb_multi_model.predict_proba(X_m_test)
    print(f"Multiclass probabilities (first 5):\n{multi_probas[:5]}")

    print("\nCatBoostClassifier tests finished.")

from catboost import CatBoostRegressor as CBRModel
import pandas as pd
from typing import Any, Dict, Optional, List

# from ..base_model import BaseModel # Zeffy: Relative import handled by consolidation order
# from ..registry import register_model # Zeffy: Relative import handled by consolidation order

@register_model("catboost_regressor")
class CatBoostRegressor(BaseModel):
    """CatBoost Regressor model."""

    def __init__(self, model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, task_type, params)
        if self.task_type != "regression":
            raise ValueError(f"CatBoostRegressor is for regression tasks, but task_type is {self.task_type}")
        
        self._default_params = {
            "objective": "RMSE", # Common objectives: RMSE, MAE, Quantile, LogLinQuantile, Poisson, MAPE
            "eval_metric": "RMSE",
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 6,
            "random_seed": self.params.get("random_state", self.params.get("random_seed", 42)),
            "verbose": 0,
            "early_stopping_rounds": None
        }
        user_params = params if params is not None else {}
        if "random_state" in user_params and "random_seed" not in user_params:
            user_params["random_seed"] = user_params.pop("random_state")

        self.params = {**self._default_params, **user_params}
        if self.params["verbose"] is None or self.params["verbose"] < 100 and self.params["verbose"] !=0:
            self.params["logging_level"] = "Silent"
        else:
            self.params["logging_level"] = "Verbose"

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, cat_features: Optional[List[int]] = None, **kwargs) -> None:
        """Trains the CatBoost regressor."""
        fit_params = self.params.copy()
        early_stopping_rounds_val = kwargs.get("early_stopping_rounds", fit_params.pop("early_stopping_rounds", None))

        self.model = CBRModel(**fit_params)
        
        eval_set_list = None
        if X_val is not None and y_val is not None:
            eval_set_list = (X_val, y_val)

        if cat_features is None:
            cat_features = [i for i, col_type in enumerate(X.dtypes) if str(col_type) == "category" or X.iloc[:, i].nunique() < self.params.get("one_hot_max", 20) and X.iloc[:,i].dtype==object]
            if cat_features:
                print(f"Inferred categorical features indices for CatBoost Regressor: {cat_features}")

        print(f"Fitting CatBoostRegressor with effective params: {self.model.get_params()}")
        try:
            self.model.fit(
                X, y, 
                eval_set=eval_set_list,
                cat_features=cat_features if cat_features else None,
                early_stopping_rounds=early_stopping_rounds_val,
            )
            self.is_trained = True
            print("CatBoostRegressor training completed.")
        except Exception as e:
            print(f"Error during CatBoostRegressor training: {e}")
            raise

    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        try:
            return self.model.predict(X, **kwargs)
        except Exception as e:
            print(f"Error during CatBoostRegressor prediction: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Optional[Any]:
        print(f"Model {self.model_type} (Regressor) does not support predict_proba.")
        return None

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_trained or not hasattr(self.model, "get_feature_importance"):
            print("Model not trained or feature importances not available for CatBoost Regressor.")
            return None
        try:
            feature_names = self.model.feature_names_ if self.model.feature_names_ else [f"feature_{i}" for i in range(len(self.model.get_feature_importance()))]
            importances = self.model.get_feature_importance()
            
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
            return importance_df
        except Exception as e:
            print(f"Could not retrieve feature importance for CatBoost Regressor: {e}")
            return None

# if __name__ == "__main__": # Zeffy: Main block from module, commented out.
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    print("Testing CatBoostRegressor...")
    X_np, y_np = make_regression(n_samples=200, n_features=20, n_informative=10, random_state=42)
    X_df = pd.DataFrame(X_np, columns=[f"feature_{i}" for i in range(X_np.shape[1])])
    X_df["feature_cat_reg_1"] = pd.Series(np.random.choice(["GroupA", "GroupB"], size=len(X_df))).astype("category")
    y_s = pd.Series(y_np)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=42)
    X_train_cb, X_val_cb, y_train_cb, y_val_cb = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    cb_reg_params = {
        "iterations": 75,
        "learning_rate": 0.07,
        "random_seed": 123,
        "objective": "RMSE",
        "eval_metric": "RMSE",
        "verbose": 0
    }
    cb_reg_model = CatBoostRegressor(model_type="catboost_regressor", task_type="regression", params=cb_reg_params)
    cb_reg_model.fit(X_train_cb, y_train_cb, X_val=X_val_cb, y_val=y_val_cb, early_stopping_rounds=10)
    
    reg_preds = cb_reg_model.predict(X_test)
    print(f"Regression predictions (first 5): {reg_preds[:5]}")
    
    rmse = mean_squared_error(y_test, reg_preds, squared=False)
    print(f"RMSE on test set: {rmse:.4f}")

    feature_imp = cb_reg_model.get_feature_importance()
    if feature_imp is not None:
        print(f"Feature importances (top 5):\n{feature_imp.head()}")

    print("\nCatBoostRegressor tests finished.")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union, Type
import importlib.util
import sys

# from ..base_model import BaseModel # Zeffy: Relative import handled by consolidation order
# from ..registry import register_model # Zeffy: Relative import handled by consolidation order

# Helper function to load a PyTorch model class from a file path
def load_pytorch_model_class_from_file(file_path: str, class_name: str) -> Type[nn.Module]:
    """Loads a PyTorch nn.Module class from a Python file."""
    try:
        spec = importlib.util.spec_from_file_location(f"custom_torch_module.{class_name}", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module at {file_path}")
        custom_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = custom_module # Add to sys.modules before exec_module
        spec.loader.exec_module(custom_module)
        model_class = getattr(custom_module, class_name)
        if not issubclass(model_class, nn.Module):
            raise TypeError(f"Class {class_name} from {file_path} is not a subclass of torch.nn.Module")
        return model_class
    except Exception as e:
        print(f"Error loading PyTorch model class 	hemed_text_color_red_bold_start_bold_end_themed_text_color_end{class_name}	hemed_text_color_red_bold_start_bold_end_themed_text_color_end from 	hemed_text_color_red_bold_start_bold_end_themed_text_color_end{file_path}	hemed_text_color_red_bold_start_bold_end_themed_text_color_end: {e}")
        raise

@register_model("pytorch_tabular_nn")
class PyTorchTabularNN(BaseModel):
    """PyTorch Neural Network model for tabular data."""

    def __init__(self, model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, task_type, params if params else {})
        
        self._default_nn_params = {
            "architecture_definition": None, # Can be a class, path to .py file, or dict for predefined
            "architecture_class_name": "CustomNet", # If loading from file
            "input_features": None, # Must be set or inferred
            "output_features": None, # Must be set or inferred (1 for regression/binary, n_classes for multiclass)
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "optimizer_name": "Adam",
            "loss_function_name": None, # Will be inferred based on task_type if None
            "device": "cpu", # or "cuda" if available and configured
            "pretrained_model_path": None,
            "fine_tune": False, # If true, only last layer might be unfrozen or specific layers
            "model_kwargs": {} # Additional kwargs for the nn.Module constructor
        }
        self.params = {**self._default_nn_params, **self.params}

        if self.params["device"] == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.params["device"] = "cpu"
        self.device = torch.device(self.params["device"])

        self.model_definition_class: Optional[Type[nn.Module]] = None
        self._initialize_model_definition()
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None

    def _initialize_model_definition(self):
        arch_def = self.params.get("architecture_definition")
        arch_class_name = self.params.get("architecture_class_name", "CustomNet")

        if isinstance(arch_def, type) and issubclass(arch_def, nn.Module):
            self.model_definition_class = arch_def
        elif isinstance(arch_def, str): # Path to a .py file
            try:
                self.model_definition_class = load_pytorch_model_class_from_file(arch_def, arch_class_name)
            except Exception as e:
                raise ValueError(f"Failed to load model architecture from file {arch_def}: {e}")
        elif isinstance(arch_def, dict): # Predefined architecture from dict (e.g. simple MLP)
            # This part can be expanded to build a model from a dictionary definition
            print("Warning: Dictionary-based architecture definition is not fully implemented yet. Please provide a class or .py file path.")
            # As a placeholder, one could define a default MLP here if arch_def specifies it
            # For now, we require a class or file.
            raise NotImplementedError("Dictionary-based architecture definition is a TODO.")
        elif arch_def is None:
            print("Warning: No architecture_definition provided. A default simple MLP will be used if input/output features are known.")
            # Define a very simple default MLP if needed, or raise error if not enough info
        else:
            raise TypeError("architecture_definition must be a nn.Module class, a path to a .py file, or a configuration dictionary.")

    def _setup_model_components(self, X: pd.DataFrame, y: pd.Series):
        if self.params["input_features"] is None:
            self.params["input_features"] = X.shape[1]
        
        if self.task_type == "classification":
            num_classes = y.nunique()
            if self.params["output_features"] is None:
                self.params["output_features"] = 1 if num_classes == 2 else num_classes
            
            if self.params["loss_function_name"] is None:
                self.criterion = nn.BCEWithLogitsLoss() if self.params["output_features"] == 1 else nn.CrossEntropyLoss()
            # Add more loss functions here (e.g., FocalLoss)
        elif self.task_type == "regression":
            if self.params["output_features"] is None:
                self.params["output_features"] = 1
            if self.params["loss_function_name"] is None:
                self.criterion = nn.MSELoss()
            # Add more loss functions (MAE, Huber, etc.)
        else:
            raise ValueError(f"Unsupported task_type for PyTorchTabularNN: {self.task_type}")

        if self.model_definition_class:
            model_kwargs = self.params.get("model_kwargs", {})
            self.model = self.model_definition_class(
                input_dim=self.params["input_features"],
                output_dim=self.params["output_features"],
                **model_kwargs
            ).to(self.device)
        elif self.params["input_features"] and self.params["output_features"]:
            # Fallback to a very simple default MLP if no architecture was provided but we have I/O dims
            print('Using default SimpleMLP with input_dim={}, output_dim={}'.format(self.params["input_features"], self.params["output_features"]))
            self.model = SimpleMLP(input_dim=self.params["input_features"], output_dim=self.params["output_features"]).to(self.device)
        else:
            raise ValueError("Cannot initialize model: input_features and output_features must be set, or a valid model_definition_class provided.")

        # Load pretrained weights if path is provided
        pretrained_path = self.params.get("pretrained_model_path")
        if pretrained_path:
            try:
                self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                print(f"Loaded pretrained model weights from {pretrained_path}")
                if self.params.get("fine_tune"):
                    # Implement fine-tuning logic (e.g., freeze some layers)
                    print("Fine-tuning enabled. (Actual layer freezing logic to be implemented)")
                    for param in self.model.parameters(): # Example: freeze all by default
                        param.requires_grad = False
                    # Unfreeze last layer or specific layers based on config
                    # e.g., if hasattr(self.model, 'fc'): for param in self.model.fc.parameters(): param.requires_grad = True 
            except Exception as e:
                print(f"Error loading pretrained model from {pretrained_path}: {e}. Training from scratch.")

        # Optimizer
        optimizer_name = self.params.get("optimizer_name", "Adam").lower()
        lr = self.params.get("learning_rate", 1e-3)
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        # Add more optimizers (AdamW, RMSprop, etc.)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, **kwargs) -> None:
        self._setup_model_components(X, y)
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("Model, optimizer, or criterion not initialized properly.")

        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        if self.task_type == "classification" and self.params["output_features"] > 1: # CrossEntropyLoss expects class indices
             y_tensor = torch.tensor(y.values, dtype=torch.long).to(self.device)
        else: # BCEWithLogitsLoss or MSELoss expect float targets
             y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(self.device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(self.device)
            if self.task_type == "classification" and self.params["output_features"] > 1:
                y_val_tensor = torch.tensor(y_val.values, dtype=torch.long).to(self.device)
            else:
                y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.params["batch_size"], shuffle=False)

        print('Starting PyTorch model training for {} epochs...'.format(self.params["epochs"]))
        self.model.train()
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            
            avg_epoch_loss = epoch_loss / len(train_loader.dataset)
            log_msg = 'Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, self.params["epochs"], avg_epoch_loss)

            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        outputs_val = self.model(batch_X_val)
                        loss_val = self.criterion(outputs_val, batch_y_val)
                        val_loss += loss_val.item() * batch_X_val.size(0)
                avg_val_loss = val_loss / len(val_loader.dataset)
                log_msg += f", Val Loss: {avg_val_loss:.4f}"
                self.model.train() # Set back to train mode
            
            if (epoch + 1) % max(1, self.params["epochs"] // 10) == 0 or epoch == self.params["epochs"] - 1:
                 print(log_msg)
        
        self.is_trained = True
        print("PyTorch model training completed.")

    def predict(self, X: pd.DataFrame, **kwargs) -> Any:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.task_type == "regression":
                return outputs.cpu().numpy().flatten()
            elif self.task_type == "classification":
                if self.params["output_features"] == 1: # Binary with sigmoid output from BCEWithLogitsLoss
                    preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int).flatten()
                else: # Multiclass with softmax output from CrossEntropyLoss
                    preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
                return preds
        return None # Should not reach here

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> Optional[Any]:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        if self.task_type != "classification":
            print("predict_proba is only available for classification tasks.")
            return None
        
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.params["output_features"] == 1: # Binary
                probas = torch.sigmoid(outputs).cpu().numpy()
                return np.hstack((1 - probas, probas)) # Return as (N, 2) array
            else: # Multiclass
                probas = torch.softmax(outputs, dim=1).cpu().numpy()
                return probas
        return None

    # save_model and load_model from BaseModel can be used if they just save/load self.model state_dict
    # Or override for more specific PyTorch saving (e.g., saving optimizer state, epoch, etc.)
    def save_model(self, path: str) -> None:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save model: not trained yet.")
        try:
            # Save the model's state_dict, architecture info might be needed for loading
            # For simplicity, just saving state_dict. Full pipeline save should handle more.
            torch.save(self.model.state_dict(), path)
            print(f"PyTorch model state_dict saved to {path}")
        except Exception as e:
            print(f"Error saving PyTorch model to {path}: {e}")
            raise

    def load_model(self, path: str) -> None:
        # This requires the model architecture to be already initialized correctly.
        # The ZeffyPipeline load mechanism should handle re-creating the model first.
        if self.model is None:
            # Attempt to re-initialize if definition is available
            if self.model_definition_class and self.params.get("input_features") and self.params.get("output_features"):
                 model_kwargs = self.params.get("model_kwargs", {})
                 self.model = self.model_definition_class(
                    input_dim=self.params["input_features"],
                    output_dim=self.params["output_features"],
                    **model_kwargs
                 ).to(self.device)
            else:
                raise RuntimeError("Cannot load model: model architecture not initialized. Call _initialize_model_definition or ensure config is set.")
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.is_trained = True
            print(f"PyTorch model state_dict loaded from {path}")
        except Exception as e:
            print(f"Error loading PyTorch model from {path}: {e}")
            raise

# A very simple default MLP for when no architecture is provided
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[List[int]] = None, activation=nn.ReLU):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [max(32, input_dim // 2), max(16, input_dim // 4)] # Simple heuristic
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation())
            layers.append(nn.Dropout(0.3))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# if __name__ == "__main__": # Zeffy: Main block from module, commented out.
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    import os

    # Create a dummy model definition file for testing file loading
    dummy_model_file_content = """
import torch.nn as nn
class CustomNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
"""
    dummy_model_path = "/home/dummy_pytorch_model.py"
    with open(dummy_model_path, "w") as f:
        f.write(dummy_model_file_content)
    print(f"Created dummy PyTorch model definition at {dummy_model_path}")

    # --- Test Classification --- 
    print("\n--- Testing PyTorchTabularNN for Classification ---")
    X_clf_np, y_clf_np = make_classification(n_samples=200, n_features=10, n_informative=5, n_classes=2, random_state=42)
    X_clf_df = pd.DataFrame(X_clf_np, columns=[f'feature_{i}' for i in range(X_clf_np.shape[1])])
    y_clf_s = pd.Series(y_clf_np)
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_clf_df, y_clf_s, test_size=0.25, random_state=42)

    nn_clf_params = {
        "architecture_definition": dummy_model_path, # Test loading from file
        "architecture_class_name": "CustomNet",
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 0.01,
        "input_features": X_c_train.shape[1], # Explicitly set for testing
        # output_features will be inferred for classification
    }
    nn_clf_model = PyTorchTabularNN(model_type="pytorch_tabular_nn", task_type="classification", params=nn_clf_params)
    nn_clf_model.fit(X_c_train, y_c_train, X_val=X_c_test, y_val=y_c_test)
    clf_preds = nn_clf_model.predict(X_c_test)
    clf_probas = nn_clf_model.predict_proba(X_c_test)
    print(f"Classification Predictions (first 5): {clf_preds[:5]}")
    print(f"Classification Probabilities (first 5):\n{clf_probas[:5]}")

    # --- Test Regression --- 
    print("\n--- Testing PyTorchTabularNN for Regression ---")
    X_reg_np, y_reg_np = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
    X_reg_df = pd.DataFrame(X_reg_np, columns=[f'feature_reg_{i}' for i in range(X_reg_np.shape[1])])
    y_reg_s = pd.Series(y_reg_np)
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_reg_df, y_reg_s, test_size=0.25, random_state=42)

    # Test with default SimpleMLP (no architecture_definition)
    nn_reg_params = {
        "epochs": 7,
        "batch_size": 16,
        "learning_rate": 0.005,
        "input_features": X_r_train.shape[1],
        "output_features": 1
    }
    nn_reg_model = PyTorchTabularNN(model_type="pytorch_tabular_nn", task_type="regression", params=nn_reg_params)
    nn_reg_model.fit(X_r_train, y_r_train, X_val=X_r_test, y_val=y_r_test)
    reg_preds = nn_reg_model.predict(X_r_test)
    print(f"Regression Predictions (first 5): {reg_preds[:5]}")

    # Clean up dummy model file
    if os.path.exists(dummy_model_path):
        os.remove(dummy_model_path)
        print(f"Removed dummy PyTorch model definition at {dummy_model_path}")

    print("\nPyTorchTabularNN tests finished.")




# --- Zeffy Pipeline Core --- 


import logging
import random
import joblib # For saving/loading pipeline

class ZeffyPipeline:
    def __init__(self, config_path: str = None, config_dict: dict = None):
        self.config = load_config(config_path=config_path, config_dict=config_dict)
        self.model_instance = None
        self.logger = self._setup_logger()
        self._apply_global_settings()
        self.preprocessors = [] 
        self.feature_engineering_steps = []

    def _setup_logger(self):
        log_level_str = self.config.global_settings.log_level.upper()
        logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO),
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            force=True) 
        logger = logging.getLogger("ZeffyPipeline")
        logger.info(f"Logger initialized for ZeffyPipeline at level {log_level_str}.")
        return logger

    def _apply_global_settings(self):
        if self.config.global_settings.random_seed is not None:
            random.seed(self.config.global_settings.random_seed)
            np.random.seed(self.config.global_settings.random_seed)
            if 'torch' in sys.modules and PYTORCH_AVAILABLE:
                 torch.manual_seed(self.config.global_settings.random_seed)
                 if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.global_settings.random_seed)
            self.logger.info(f"Global random seed set to: {self.config.global_settings.random_seed}")

    def _load_data(self):
        self.logger.info(f"Loading data using: {self.config.data_loader.type}")
        if self.config.data_loader.type == "sklearn_dataset":
            if self.config.data_loader.dataset_name == "iris":
                from sklearn.datasets import load_iris
                data = load_iris()
                X = pd.DataFrame(data.data, columns=data.feature_names)
                y = pd.Series(data.target, name=self.config.data_loader.target_column or "target")
                return X, y
            else:
                raise ZeffyDataError(f"Sklearn dataset '{self.config.data_loader.dataset_name}' not implemented in example loader.")
        elif self.config.data_loader.type == "csv":
            if not self.config.data_loader.path:
                raise ZeffyDataError("CSV path not specified in data_loader config.")
            if not os.path.exists(self.config.data_loader.path):
                raise ZeffyDataError(f"CSV file not found at: {self.config.data_loader.path}")
            df = pd.read_csv(self.config.data_loader.path)
            if not self.config.data_loader.target_column:
                raise ZeffyDataError("Target column not specified for CSV data_loader.")
            if self.config.data_loader.target_column not in df.columns:
                raise ZeffyDataError(f"Target column '{self.config.data_loader.target_column}' not found in CSV.")
            X = df.drop(columns=[self.config.data_loader.target_column])
            y = df[self.config.data_loader.target_column]
            return X,y
        raise ZeffyDataError(f"Data loader type '{self.config.data_loader.type}' not implemented.")

    def fit(self, X_train=None, y_train=None, X_val=None, y_val=None):
        self.logger.info(f"Starting Zeffy pipeline fitting for project: {self.config.project_name}")
        if X_train is None and y_train is None:
            X_train, y_train = self._load_data()
            self.logger.info(f"Data loaded internally: X_train shape {X_train.shape if X_train is not None else 'None'}, y_train shape {y_train.shape if y_train is not None else 'None'}")
        elif X_train is None or y_train is None:
            raise ZeffyDataError("Both X_train and y_train must be provided, or neither (to use internal data loader).")
        X_train_featured = X_train 
        X_val_featured = X_val 
        if not self.config.models:
            raise ZeffyConfigurationError("No models specified in the configuration.")
        model_config = self.config.models[0]
        self.logger.info(f"Preparing model: {model_config.type} with task: {model_config.task_type}")
        model_params_dict = model_config.params.model_dump() if model_config.params else {}
        if 'random_state' not in model_params_dict and 'random_seed' not in model_params_dict and self.config.global_settings.random_seed is not None:
             if model_config.type in ["lightgbm_classifier", "lightgbm_regressor", "xgboost_classifier", "xgboost_regressor", "sklearn_logistic_regression"]:
                 model_params_dict['random_state'] = self.config.global_settings.random_seed
             elif model_config.type in ["catboost_classifier", "catboost_regressor"]:
                 model_params_dict['random_seed'] = self.config.global_settings.random_seed
        self.model_instance = get_model(model_type=model_config.type, 
                                        task_type=model_config.task_type, 
                                        params=model_params_dict)
        self.logger.info(f"Fitting model {model_config.type}...")
        self.model_instance.fit(X_train_featured, y_train, X_val=X_val_featured, y_val=y_val)
        self.logger.info("Model training completed.")

    def predict(self, X_test):
        if not self.model_instance or not self.model_instance.is_trained:
            raise ZeffyNotFittedError("Pipeline has not been fitted or model is not trained.")
        X_test_featured = X_test
        self.logger.info("Making predictions.")
        return self.model_instance.predict(X_test_featured)

    def predict_proba(self, X_test):
        if not self.model_instance or not self.model_instance.is_trained:
            raise ZeffyNotFittedError("Pipeline has not been fitted or model is not trained.")
        model_config = self.config.models[0]
        if model_config.task_type != "classification":
            self.logger.warning("predict_proba is for classification models.")
            return None
        X_test_featured = X_test
        self.logger.info("Making probability predictions.")
        return self.model_instance.predict_proba(X_test_featured)

    def evaluate(self, X_test, y_test, metrics=None):
        from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, roc_auc_score
        self.logger.info("Evaluating model.")
        model_config = self.config.models[0]
        task_type = model_config.task_type
        results = {}
        if metrics is None:
            metrics = self.config.evaluation.metrics if self.config.evaluation else []
        if not metrics:
            metrics = ["accuracy"] if task_type == "classification" else ["rmse"]
            self.logger.warning(f"No metrics specified for evaluation, using default: {metrics}")
        predictions = self.predict(X_test)
        for metric_name in metrics:
            try:
                if task_type == "classification":
                    if metric_name == "accuracy":
                        results[metric_name] = accuracy_score(y_test, predictions)
                    elif metric_name == "f1_macro":
                        results[metric_name] = f1_score(y_test, predictions, average="macro")
                    elif metric_name == "roc_auc_ovr" or metric_name == "roc_auc":
                        probas = self.predict_proba(X_test)
                        if probas is not None:
                            if probas.shape[1] == 2:
                                results[metric_name] = roc_auc_score(y_test, probas[:, 1])
                            else:
                                results[metric_name] = roc_auc_score(y_test, probas, multi_class="ovr", average="macro")
                elif task_type == "regression":
                    if metric_name == "rmse":
                        results[metric_name] = mean_squared_error(y_test, predictions, squared=False)
                    elif metric_name == "mae":
                        from sklearn.metrics import mean_absolute_error
                        results[metric_name] = mean_absolute_error(y_test, predictions)
                    elif metric_name == "r2":
                        from sklearn.metrics import r2_score
                        results[metric_name] = r2_score(y_test, predictions)
                self.logger.info(f"Calculated metric {metric_name}: {results.get(metric_name)}")
            except Exception as e:
                self.logger.error(f"Could not calculate metric {metric_name}: {e}")
                results[metric_name] = None
        self.logger.info(f"Evaluation results: {results}")
        return results

    def save_pipeline(self, path_prefix: str):
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix, exist_ok=True)
        pipeline_file = os.path.join(path_prefix, "zeffy_pipeline.joblib")
        config_file = os.path.join(path_prefix, "zeffy_config.yaml")
        try:
            temp_logger = self.logger
            self.logger = None 
            joblib.dump(self, pipeline_file)
            self.logger = temp_logger
            with open(config_file, 'w') as f:
                yaml.dump(self.config.model_dump(mode='json'), f)
            self.logger.info(f"Pipeline saved to {path_prefix}")
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                 self.logger.error(f"Error saving pipeline: {e}")
            else:
                 print(f"Error saving pipeline (logger not available): {e}")
            raise ZeffyError(f"Failed to save pipeline to {path_prefix}: {e}")

    @staticmethod
    def load_pipeline(path_prefix: str):
        pipeline_file = os.path.join(path_prefix, "zeffy_pipeline.joblib")
        if not os.path.exists(pipeline_file):
            raise ZeffyError(f"Pipeline file not found at {pipeline_file}")
        try:
            pipeline = joblib.load(pipeline_file)
            pipeline.logger = pipeline._setup_logger()
            pipeline.logger.info(f"Pipeline loaded from {path_prefix}")
            return pipeline
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise ZeffyError(f"Failed to load pipeline from {path_prefix}: {e}")


# --- Example Usage for Consolidated zeffy.py ---
if __name__ == "__main__":
    print("Running Zeffy consolidated example...")
    dummy_config_dict = {
        "project_name": "Zeffy Single File Test",
        "global_settings": {"random_seed": 42, "log_level": "INFO"},
        "data_loader": {
            "type": "sklearn_dataset",
            "dataset_name": "iris",
            "target_column": "target" 
        },
        "models": [
            {
                "type": "lightgbm_classifier", 
                "task_type": "classification",
                "params": {"n_estimators": 20, "learning_rate": 0.1}
            }
        ],
        "evaluation": {
            "metrics": ["accuracy", "f1_macro"]
        }
    }
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris_data = load_iris()
    X_main = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    y_main = pd.Series(iris_data.target, name="target")
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(X_main, y_main, test_size=0.3, random_state=42, stratify=y_main)
    dummy_config_path = "/home/ubuntu/dummy_zeffy_config.yaml"
    try:
        with open(dummy_config_path, 'w') as f:
            yaml.dump(dummy_config_dict, f)
        
        pipeline_from_file = ZeffyPipeline(config_path=dummy_config_path)
        pipeline_from_file.fit(X_train=X_train_main, y_train=y_train_main)

        print("--- Making predictions ---")
        predictions = pipeline_from_file.predict(X_test_main)
        print(f"Predictions (first 10): {predictions[:10]}")

        if dummy_config_dict['models'][0]['task_type'] == "classification":
            probabilities = pipeline_from_file.predict_proba(X_test_main)
            if probabilities is not None:
                print(f"Probabilities (first 5):\n{probabilities[:5]}")

        print("--- Evaluating pipeline ---")
        eval_results = pipeline_from_file.evaluate(X_test_main, y_test_main)
        print(f"Evaluation results: {eval_results}")

        print("--- Saving pipeline ---")
        save_path = "/home/ubuntu/zeffy_saved_pipeline"
        pipeline_from_file.save_pipeline(save_path)
        print(f"Pipeline saved to {save_path}")

        print("--- Loading pipeline ---")
        loaded_pipeline = ZeffyPipeline.load_pipeline(save_path)
        print(f"Pipeline loaded. Project: {loaded_pipeline.config.project_name}")
        loaded_predictions = loaded_pipeline.predict(X_test_main)
        
        if np.array_equal(predictions, loaded_predictions):
            print("Loaded pipeline predictions match original. Test successful!")
        else:
            print("Loaded pipeline predictions DO NOT match. Test FAILED.")
    except Exception as e:
        print(f"Error in Zeffy example: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path) # remove directory and its contents

    print("Zeffy consolidated example finished.")
