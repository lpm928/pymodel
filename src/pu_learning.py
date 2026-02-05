import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# === FeatureWeighter Class Definition ===
class FeatureWeighter(BaseEstimator, TransformerMixin):
    """
    Apply weights to specific features after preprocessing.
    """
    def __init__(self, preprocess, weight_dict=None):
        self.preprocess = preprocess
        self.weight_dict = weight_dict if weight_dict is not None else {}
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        # We need to fit the preprocessor first to get feature names
        try:
             # Check if already fitted? usually preprocess is fitted in pipeline before this
             # But here FeatureWeighter WRAPS preprocess if passed as arg?
             # Based on user code: ('weighter', FeatureWeighter(preprocess=preprocess...))
             # So this class takes the preprocessor and acts as the transformer.
             self.preprocess.fit(X)
        except:
             pass 
             
        # Extract feature names
        # This part depends heavily on the structure of 'preprocess' (ColumnTransformer)
        
        # Helper to get names from ColumnTransformer
        output_features = []
        
        if hasattr(self.preprocess, 'transformers_'):
            for name, trans, column_names in self.preprocess.transformers_:
                if name == 'remainder' and trans == 'drop':
                    continue
                    
                if hasattr(trans, 'get_feature_names_out'):
                    # For OneHotEncoder etc
                    names = trans.get_feature_names_out(column_names)
                    output_features.extend(names)
                elif hasattr(trans, 'steps'): # Pipeline
                    # Assume last step has feature names or just use cols
                    # For scaler/imputer pipelines, feature names are usually preserved 1-to-1 if not OHE
                    # But sklearn pipelines don't always propagate names easily.
                    # We will assume 1-to-1 mapping for numerical pipelines
                    output_features.extend(column_names)
                else:
                    output_features.extend(column_names)
        
        self.feature_names_out_ = np.array(output_features)
        return self

    def transform(self, X):
        Xt = self.preprocess.transform(X)
        Xt_dense = Xt.toarray() if hasattr(Xt, 'toarray') else Xt
        
        # If feature names extraction failed or length mismatch, fallback to index?
        # User code assumes strict matching. Let's try to be robust.
        if self.feature_names_out_ is not None and len(self.feature_names_out_) == Xt_dense.shape[1]:
            df = pd.DataFrame(Xt_dense, columns=self.feature_names_out_)
            
            for orig_col, w in self.weight_dict.items():
                # Direct match
                if orig_col in df.columns:
                     df[orig_col] = df[orig_col] * w
                
                # Prefix match (for One-Hot)
                cols_to_weight = [c for c in df.columns if c.startswith(orig_col + '_')]
                for c in cols_to_weight:
                    df[c] = df[c] * w
            
            return df.values
        
        return Xt_dense

def train_pu_model(pos_df, unlabeled_df, feature_weights=None):
    """
    Train PU Model given Positives (A) and Unlabeled (B).
    Returns: (fitted_calibrated_model, metrics_dict, pipeline_for_feature_names)
    """
    
    # 1. Labeling
    pos = pos_df.copy()
    pos['label'] = 1
    
    # Sample B to match A size (balanced PU)
    # If B is smaller than A, take all B
    n_sample = min(len(pos), len(unlabeled_df))
    neg = unlabeled_df.sample(n=n_sample, replace=False, random_state=42).copy()
    neg['label'] = 0
    
    train_df = pd.concat([pos, neg], axis=0).reset_index(drop=True)
    
    # 2. Identify Columns (Auto-detect based on data types)
    # We exclude 'label' and 'Batch_ID' and 'ID' columns
    exclude_cols = ['label', 'Batch_ID']
    # Try to find ID col from metadata or heuristic? 
    # For now, drop non-numeric/common ID names if present
    
    features = [c for c in train_df.columns if c not in exclude_cols and 'ID' not in c and '編號' not in c]
    
    # Split Num/Cat
    # Simple heuristic: object/category = Cat, number = Num
    X = train_df[features]
    y = train_df['label'].astype(int)
    
    features_num = X.select_dtypes(include=[np.number]).columns.tolist()
    features_cat = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # 3. Pipelines
    # Log Transform cols: usually financial amounts. 
    # For general purpose, let's log-transform all Num features that have high variance?
    # Or just use standard scaler. User code separated them. 
    # For generic implementation, we use Standard Scaler for all.
    
    num_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for easier weighting
    ])
    
    transformers = []
    if features_num:
        transformers.append(('num', num_tf, features_num))
    if features_cat:
        transformers.append(('cat', cat_tf, features_cat))
        
    preprocess = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # 4. Weighter
    if feature_weights is None:
        feature_weights = {}
        
    # 5. Classifier
    base_clf = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        C=10.0, # Slightly stronger regularization than 100
        random_state=42
    )
    
    # Pipeline
    # Note: FeatureWeighter wrapping preprocess
    main_pipe = Pipeline(steps=[
        ('weighter', FeatureWeighter(preprocess=preprocess, weight_dict=feature_weights)),
        ('clf', base_clf)
    ])
    
    # Fit Main Pipe (to get inner feature names and internal state)
    main_pipe.fit(X, y)
    
    # 6. Calibration
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    calibrated_clf = CalibratedClassifierCV(estimator=main_pipe, method='sigmoid', cv=cv)
    calibrated_clf.fit(X, y)
    
    # Eval
    probs = main_pipe.predict_proba(X)[:, 1] # Use inner pipe to check training fit
    auc = roc_auc_score(y, probs)
    
    metrics = {
        "auc": auc,
        "pos_samples": len(pos),
        "neg_samples_used": len(neg)
    }
    
    return calibrated_clf, metrics, main_pipe

