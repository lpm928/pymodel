import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score
from datetime import datetime

MODEL_DIR = "d:/AI/Antigravity/SKB/models"
os.makedirs(MODEL_DIR, exist_ok=True)

class ModelEngine:
    def __init__(self):
        pass
    
    def _prepare_data(self, df, target_col=None, id_col=None):
        """Separate features (X) and target (y), dropping ID if present."""
        df_mod = df.copy()
        
        # Drop known non-feature columns
        cols_to_drop = ["Batch_ID"]
        if id_col and id_col in df_mod.columns:
            cols_to_drop.append(id_col)
            
        # Initial drop
        df_mod = df_mod.drop(columns=[c for c in cols_to_drop if c in df_mod.columns])
        
        y = None
        if target_col and target_col in df_mod.columns:
            y = df_mod[target_col]
            df_mod = df_mod.drop(columns=[target_col])
            
        # Critical: Filter only valid feature types (Numeric/Bool)
        # This removes "Unused" string columns that cleaner.py might have left behind.
        X = df_mod.select_dtypes(include=[np.number, bool])
        
        return X, y

    def train_lookalike(self, df, id_col=None):
        """
        Train a Lookalike Model (One-Class) using Isolation Forest.
        Assumption: df contains only POSITIVE examples (purchasers).
        """
        X, _ = self._prepare_data(df, target_col=None, id_col=id_col)
        
        # Isolation Forest: outliers are different from training data. 
        # Since training data is "Purchasers", high score = similar to Purchasers.
        # Contamination='auto' or low value implies training data is mostly pure.
        model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
        model.fit(X)
        
        # We can't really return accuracy metrics for one-class without negatives.
        # We can return 'num_samples' or simple stats.
        metrics = {"num_samples": len(X), "features": len(X.columns)}
        return model, metrics

    def train_classification(self, df, target_col, id_col=None, model_type='rf'):
        """Train a classification model (Workflow 1)."""
        X, y = self._prepare_data(df, target_col, id_col)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = GradientBoostingClassifier(random_state=42)
            
        model.fit(X_train, y_train)
        
        # Eval
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return model, report

    def train_regression(self, df, target_col, id_col=None):
        """Train a regression model (Workflow 3)."""
        X, y = self._prepare_data(df, target_col, id_col)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {"mse": mse, "r2": r2}
        return model, metrics

    def train_clustering(self, df, id_col=None, k=None):
        """Train a clustering model (Workflow 2)."""
        X, _ = self._prepare_data(df, target_col=None, id_col=id_col)
        
        # If K is not provided, use simple Elbow-like heuristic or default
        if k is None:
            # Simple logic: try 3 to 8, pick best silhouette
            best_model = None
            best_score = -1
            best_k = 3
            
            # Limit K search to avoid performance hit on large data
            search_range = range(3, min(10, len(X)))
            
            for curr_k in search_range:
                model = KMeans(n_clusters=curr_k, random_state=42, n_init='auto')
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_k = curr_k
            
            model = best_model
            labels = model.labels_
            metrics = {"k": best_k, "silhouette_score": best_score}
        else:
            model = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            metrics = {"k": k, "silhouette_score": score}
            
        # Return labels so we can append to DF immediately
        return model, metrics, labels

    def predict(self, model, df, id_col=None):
        """Generic predict."""
        X, _ = self._prepare_data(df, target_col=None, id_col=id_col)
        
        # Check if model has predict_proba (Classifier)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1] # Probability of class 1
            preds = model.predict(X)
            return preds, probs
        elif isinstance(model, IsolationForest):
            # Lookalike Logic
            # decision_function: average anomaly score of X of the base classifiers.
            # Higher is better (more normal/similar to training data).
            # Output range is roughly -0.5 to 0.5 depending on implementation.
            raw_scores = model.decision_function(X)
            
            # Normalize to 0-1 probability-like score
            # We use MinMaxScaler logic but fitted on these batches. 
            # Ideally we should store min/max from training, but IsolationForest boundaries are dynamic.
            # A sigmoid or simple min-max on the batch is a practical approximation for ranking.
            
            # Simple MinMax for this batch to rank them:
            min_s = raw_scores.min()
            max_s = raw_scores.max()
            if max_s - min_s == 0:
                probs = np.ones(len(raw_scores)) * 0.5
            else:
                probs = (raw_scores - min_s) / (max_s - min_s)
            
            # Preds: 1 if positive score (inlier), -1 if negative (outlier)
            # We map -1 to 0 (Low Potential) and 1 to 1 (High Potential)
            orig_preds = model.predict(X)
            preds = np.where(orig_preds == 1, 1, 0)
            
            return preds, probs
        else:
            # Regression or Clustering
            preds = model.predict(X)
            return preds, None

    def update_model(self, current_model, df_new, target_col=None, id_col=None, model_type="lookalike"):
        """
        Feedback Loop (Mode C).
        Refit model with new data (Incremental not fully supported by RF/IF, so we combine or retrain).
        For simplicity in this MVP: We assume 'df_new' is the *cumulative* dataset or we just train on new batch?
        Ideally: Users upload 'Actual Results' of previous predictions.
        
        Scenario:
        1. Lookalike: User uploads more "Successful" customers. -> Train new IF on (Old + New). 
           (But we don't have Old here easily unless stored. For MVP, we assume User uploads NEW combined list or just refits on new strong signals).
           Let's assume: User uploads "Actual Purchasers from Prediction List". We retrain a model on this "Confirmed High Value" group.
           
        2. Classification: User uploads (Features + Actual Target). Refit classifier.
        """
        
        # For this phase, we'll treat "Update" as "Training a new version v1.1" using the provided feedback data.
        # If user wants to merge with old data, they should ideally handle data merging outside or we provide a merge tool.
        # But commonly, "Retraining on latest ground truth" is a good step.
        
        if isinstance(current_model, IsolationForest) or model_type == "lookalike":
            # Train new Lookalike model on this new positive feedback data
            return self.train_lookalike(df_new, id_col)
            
        elif target_col:
            # Train new Classification/Regression model
            # Detect type by model
            if isinstance(current_model, (RandomForestClassifier, GradientBoostingClassifier)):
                 return self.train_classification(df_new, target_col, id_col)
            elif isinstance(current_model, (RandomForestRegressor, GradientBoostingRegressor)):
                 return self.train_regression(df_new, target_col, id_col)
        
        raise ValueError("Unknown model type or missing target column for update.")

    def save_model(self, model, name_prefix="model"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name_prefix}_v{timestamp}.joblib"
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(model, path)
        return path, filename

    def load_model(self, path):
        return joblib.load(path)

    def list_models(self):
        """List all saved models sorted by recentness."""
        if not os.path.exists(MODEL_DIR):
            return []
        
        files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]
        # Sort by creation time (or filename timestamp if strictly followed, creation time is safer if user renames)
        # But filename sort is good enough for vYYYYMMDD...
        files.sort(reverse=True)
        return files
