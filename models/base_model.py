"""
Base model interface for the SMU-Textron Cognitive Load dataset analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(ABC):
    """Abstract base class for models."""
    
    @abstractmethod
    def train(self, X_train, y_train) -> None:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X) -> Any:
        """Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass


"""
TODO Improvements:
1. Add methods for model serialization (save/load)
2. Implement abstract methods for feature importance
3. Add methods for cross-validation
4. Implement methods for hyperparameter tuning
5. Add support for tracking training progress
6. Implement early stopping functionality
7. Add methods for model comparison
8. Implement confidence intervals for predictions
9. Add support for model ensembling
10. Implement methods for model interpretability
11. Add support for online learning
"""
