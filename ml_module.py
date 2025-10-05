"""
Machine Learning Module for Sports Betting Predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import shap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLModel:
    """Machine Learning model for sports betting predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.calibration_model = None
        self.shap_explainer = None
        
    def prepare_features(self, game_data):
        """Prepare features from game data"""
        features = []
        feature_names = []
        
        # Basic game features
        if isinstance(game_data, dict):
            # Single game data
            features.extend([
                game_data.get('spread', 0) or 0,
                game_data.get('total', 0) or 0,
                game_data.get('ml_away', 100) or 100,
                game_data.get('ml_home', -100) or -100,
                game_data.get('spread_vig', -110) or -110,
                game_data.get('total_vig', -110) or -110
            ])
            feature_names.extend(['spread', 'total', 'ml_away', 'ml_home', 'spread_vig', 'total_vig'])
            
            # Implied probabilities
            ml_away_prob = implied_probability(game_data.get('ml_away', 100))
            ml_home_prob = implied_probability(game_data.get('ml_home', -100))
            features.extend([ml_away_prob, ml_home_prob])
            feature_names.extend(['implied_prob_away', 'implied_prob_home'])
            
            # No-vig probabilities if available
            if game_data.get('spread_no_vig'):
                features.append(game_data['spread_no_vig'].get('prob1', 0.5))
                feature_names.append('spread_no_vig_prob')
            else:
                features.append(0.5)
                feature_names.append('spread_no_vig_prob')
                
            if game_data.get('total_no_vig'):
                features.append(game_data['total_no_vig'].get('prob1', 0.5))
                feature_names.append('total_no_vig_prob')
            else:
                features.append(0.5)
                feature_names.append('total_no_vig_prob')
                
            if game_data.get('ml_no_vig'):
                features.append(game_data['ml_no_vig'].get('prob1', 0.5))
                feature_names.append('ml_no_vig_prob')
            else:
                features.append(0.5)
                feature_names.append('ml_no_vig_prob')
        
        return np.array(features).reshape(1, -1), feature_names
    
    def fit(self, X, y):
        """Train the model with time-series cross-validation"""
        if len(np.unique(y)) < 2:
            # Not enough classes, use dummy model
            self.model = DummyModel()
            self.model.fit(X, y)
            self.is_trained = True
            return
        
        # Use LightGBM for better performance
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=tscv, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model
        self.model.fit(X, y)
        
        # Calibrate probabilities
        self.calibration_model = CalibratedClassifierCV(self.model, cv='prefit', method='isotonic')
        self.calibration_model.fit(X, y)
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        self.is_trained = True
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    def predict_proba(self, X):
        """Predict probabilities with calibration"""
        if not self.is_trained:
            return np.array([[0.5, 0.5]] * len(X))
        
        if self.calibration_model:
            return self.calibration_model.predict_proba(X)
        else:
            return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels"""
        if not self.is_trained:
            return np.zeros(len(X))
        return self.model.predict(X)
    
    def get_shap_values(self, X):
        """Get SHAP values for explanation"""
        if not self.is_trained or self.shap_explainer is None:
            return None
        return self.shap_explainer.shap_values(X)

class DummyModel:
    """Dummy model for when there's insufficient data"""
    def __init__(self):
        self.is_trained = True
        
    def fit(self, X, y):
        pass
        
    def predict_proba(self, X):
        return np.array([[0.5, 0.5]] * len(X))
        
    def predict(self, X):
        return np.zeros(len(X))

def implied_probability(odds):
    """Calculate implied probability from moneyline odds"""
    if odds == 'even':
        return 0.5
    try:
        odds_val = float(odds)
        if odds_val > 0:
            return 100 / (odds_val + 100)
        elif odds_val < 0:
            return abs(odds_val) / (abs(odds_val) + 100)
        else:
            return 0.5
    except (ValueError, TypeError):
        return 0.5

def create_training_data(historical_games):
    """Create training data from historical games"""
    if not historical_games or len(historical_games) < 10:
        return None, None
    
    features = []
    targets = []
    
    for i in range(1, len(historical_games)):
        current_game = historical_games[i]
        previous_game = historical_games[i-1]
        
        # Create features from current and previous game state
        feature_vector = []
        
        # Current game features
        feature_vector.extend([
            current_game.get('spread', 0) or 0,
            current_game.get('total', 0) or 0,
            current_game.get('ml_away', 100) or 100,
            current_game.get('ml_home', -100) or -100,
        ])
        
        # Implied probabilities
        feature_vector.append(implied_probability(current_game.get('ml_away', 100)))
        feature_vector.append(implied_probability(current_game.get('ml_home', -100)))
        
        # Changes from previous game
        feature_vector.extend([
            (current_game.get('spread', 0) or 0) - (previous_game.get('spread', 0) or 0),
            (current_game.get('total', 0) or 0) - (previous_game.get('total', 0) or 0),
            (current_game.get('ml_away', 100) or 100) - (previous_game.get('ml_away', 100) or 100),
        ])
        
        # Target: whether the underdog covered (simplified)
        # In a real implementation, you would use actual game outcomes
        spread_change = (current_game.get('spread', 0) or 0) - (previous_game.get('spread', 0) or 0)
        target = 1 if spread_change < 0 else 0  # Simplified target
        
        features.append(feature_vector)
        targets.append(target)
    
    return np.array(features), np.array(targets)

def train_model(historical_data):
    """Train machine learning model on historical data"""
    model = MLModel()
    
    # Create training data
    X, y = create_training_data(historical_data)
    
    if X is not None and len(X) > 0:
        # Scale features
        X_scaled = model.scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        # Save model
        joblib.dump(model, 'sports_betting_model.joblib')
    
    return model

def predict_game(model, game_data):
    """Predict game outcome probability"""
    if not model.is_trained:
        return 0.5, 0.5, None
    
    try:
        # Prepare features
        X, feature_names = model.prepare_features(game_data)
        X_scaled = model.scaler.transform(X)
        
        # Predict probabilities
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Calculate confidence based on probability distribution
        confidence = 1 - (2 * abs(probabilities[0] - 0.5))
        
        # Get SHAP values for explanation
        shap_values = model.get_shap_values(X_scaled)
        
        return probabilities[1], confidence, shap_values
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.5, 0.5, None

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    if not model.is_trained:
        return {"accuracy": 0.5, "log_loss": 1.0}
    
    X_test_scaled = model.scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    
    return {
        "accuracy": accuracy,
        "log_loss": loss
    }

# Model persistence functions
def save_model(model, filepath):
    """Save trained model to file"""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load trained model from file"""
    return joblib.load(filepath)
