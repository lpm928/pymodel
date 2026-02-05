import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score
from datetime import datetime
try:
    from src import pu_learning
except ImportError:
    import pu_learning # Local fallback

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

    def train_pu_learning(self, pos_df, unlabeled_df, feature_weights=None, id_col=None):
        """
        Train PU Model (Workflow 4) using src/pu_learning.py
        """
        # We need to make sure we don't pass ID columns to the training logic
        # pu_learning.train_pu_model handles dropping, but let's be safe
        model, metrics, pipe = pu_learning.train_pu_model(pos_df, unlabeled_df, feature_weights)
        return model, metrics

    def predict(self, model, df, id_col=None):
        """Generic predict."""
        # For PU Learning (CalibratedClassifierCV), the input to predict must be the raw DF 
        # because the internal pipeline handles preprocessing (ColumnTransformer).
        # Standard models (RF/GBM) expect pre-processed X from _prepare_data.
        
        # Check model type
        is_pu = isinstance(model, (pu_learning.CalibratedClassifierCV, pu_learning.Pipeline)) or \
                (hasattr(model, 'estimator') and hasattr(model.estimator, 'steps') and 'weighter' in model.estimator.named_steps)

        if is_pu:
             # PU Model expects DataFrame with columns matching training time
             # We should still probably drop ID if it's not part of features?
             # But pipeline usually filters logic. Let's pass DF but maybe drop ID if we know it.
             # Actually pu_learning logic selects features using exclusion list, so passing raw DF including ID is 'okay' 
             # IF the exclusion list covered it. But let's be safe and try to drop ID if provided.
             df_input = df.copy()
             if id_col and id_col in df_input.columns:
                 df_input.drop(columns=[id_col], inplace=True)
             
             probs = model.predict_proba(df_input)[:, 1]
             # Decile logic or threshold?
             # Return probs as score. Preds > 0.5?
             preds = (probs > 0.5).astype(int)
             return preds, probs

        # Standard Logic
        X, _ = self._prepare_data(df, target_col=None, id_col=id_col)
        
        # Check if model has predict_proba (Classifier)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1] # Probability of class 1
            preds = model.predict(X)
            return preds, probs
        elif isinstance(model, IsolationForest):
            # Lookalike Logic
            raw_scores = model.decision_function(X)
            min_s = raw_scores.min()
            max_s = raw_scores.max()
            if max_s - min_s == 0:
                probs = np.ones(len(raw_scores)) * 0.5
            else:
                probs = (raw_scores - min_s) / (max_s - min_s)
            
            orig_preds = model.predict(X)
            preds = np.where(orig_preds == 1, 1, 0)
            
            return preds, probs
        else:
            # Regression or Clustering
            preds = model.predict(X)
            return preds, None

    def update_model(self, current_model, df_new, target_col=None, id_col=None, model_type="standard"):
        """
        Feedback Loop (Mode C).
        """
        if isinstance(current_model, IsolationForest) or model_type == "lookalike":
            return self.train_lookalike(df_new, id_col)
            
        elif model_type == "pu_learning":
             # For PU, df_new usually means "New Positives" (Confirmed Purchases).
             # But we also need Unlabeled data to retrain PU!
             # This is tricky without persisting the original Unlabeled set.
             # MVP: Assume df_new contains BOTH Positives and some Unlabeled? 
             # OR: Raise error that PU requires full re-training in Tab 1?
             # Let's assume user uploads a MERGED file of (Old Positives + New Positives).
             # And we need Unlabeled data... 
             # Hack: We can't easily update PU without Unlabeled.
             # Let's Skip PU update in this simple loop or request B-set.
             # Better: Allow User to upload "New Positives", and we mix it with a synthetic B set?
             # For now, return error or handle gracefully.
             raise NotImplementedError("PU Learning update requires both Positive and Unlabeled data. Please retrain in Tab 2.")

        elif target_col:
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
        files.sort(reverse=True)
        return files
