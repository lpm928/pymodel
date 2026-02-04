import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import uuid

def clean_data(df, metadata, options={}):
    """
    Main pipeline for cleaning and feature engineering.
    
    Args:
        df: Input pandas DataFrame
        metadata: Dictionary mapping column names to types: 
                  ('ID', 'Numerical', 'Categorical', 'Target', 'Datetime')
        options: Dictionary for configuration options (scaling method, imputation strategy, etc.)
        
    Returns:
        Processed DataFrame
    """
    df_clean = df.copy()
    
    # 0. Batch Tracking
    if 'Batch_ID' not in df_clean.columns:
        batch_id = options.get('batch_id', str(uuid.uuid4())[:8])
        df_clean['Batch_ID'] = batch_id
    
    # Drop duplicates
    if options.get('drop_duplicates', True):
        df_clean.drop_duplicates(inplace=True)

    # Separate columns by type
    numerical_cols = [col for col, type_ in metadata.items() if type_ == 'Numerical' and col in df_clean.columns]
    categorical_cols = [col for col, type_ in metadata.items() if type_ == 'Categorical' and col in df_clean.columns]
    datetime_cols = [col for col, type_ in metadata.items() if type_ == 'Datetime' and col in df_clean.columns]
    target_col = next((col for col, type_ in metadata.items() if type_ == 'Target'), None)

    # 1. Missing Values Imputation
    # 1. Missing Values Imputation
    # Numerical Pre-processing (Handle commas, spaces)
    if numerical_cols:
        for col in numerical_cols:
            if df_clean[col].dtype == object or df_clean[col].dtype == str:
                # Remove commas and strip whitespace
                df_clean[col] = df_clean[col].astype(str).str.replace(',', '').str.strip()
                # Force to numeric, coerce errors to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Numerical Imputation
    num_strategy = options.get('numerical_impute_strategy', 'median')
    if numerical_cols:
        num_imputer = SimpleImputer(strategy=num_strategy)
        # Check if all columns are all-NaN (which causes imputer error)
        valid_num_cols = [c for c in numerical_cols if not df_clean[c].isna().all()]
        if valid_num_cols:
            df_clean[valid_num_cols] = num_imputer.fit_transform(df_clean[valid_num_cols])
    
    # Categorical
    cat_strategy = options.get('categorical_impute_strategy', 'constant')
    fill_value = options.get('categorical_fill_value', 'Unknown')
    if categorical_cols:
        if cat_strategy == 'constant':
            df_clean[categorical_cols] = df_clean[categorical_cols].fillna(fill_value)
        else:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols] = cat_imputer.fit_transform(df_clean[categorical_cols])

    # 2. Outlier Handling (Numerical)
    if options.get('handle_outliers', False) and numerical_cols:
        method = options.get('outlier_method', 'iqr')
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Clip outliers instead of removing to preserve data shape? 
                # Or remove rows? Let's cap them for now to preserve data.
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'zscore':
                # Simplified z-score clipping
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 0:
                    df_clean[col] = df_clean[col].clip(lower=mean-3*std, upper=mean+3*std)

    # 3. Dynamic Feature Engineering - Time Series
    for col in datetime_cols:
        # Ensure datetime type
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Extract features
        prefix = col
        df_clean[f'{prefix}_month'] = df_clean[col].dt.month
        df_clean[f'{prefix}_day_of_week'] = df_clean[col].dt.dayofweek
        df_clean[f'{prefix}_is_weekend'] = (df_clean[col].dt.dayofweek >= 5).astype(int)
        
        # Calculate days since today (recency)
        today = pd.Timestamp.now()
        df_clean[f'{prefix}_days_since'] = (today - df_clean[col]).dt.days

        # Drop original datetime col if requested (usually good for ML models)
        if options.get('drop_original_datetime', True):
            df_clean.drop(columns=[col], inplace=True)

    # 4. Scaling (Numerical)
    scale_method = options.get('scaling_method', 'standard')
    if numerical_cols:
        if scale_method == 'standard':
            scaler = StandardScaler()
            df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])
        elif scale_method == 'minmax':
            scaler = MinMaxScaler()
            df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

    # 5. Encoding (Categorical)
    # Check cardinality to decide strategy
    high_cardinality_threshold = options.get('cardinality_threshold', 20)
    
    for col in categorical_cols:
        unique_count = df_clean[col].nunique()
        encoding_type = options.get(f'encoding_{col}', 'auto')
        
        if encoding_type == 'auto':
            if unique_count > high_cardinality_threshold:
                encoding_type = 'frequency' # Default to frequency for high cardinality if target not avail yet
                if target_col and options.get('use_target_encoding', False):
                    encoding_type = 'target'
            else:
                encoding_type = 'onehot'
        
        if encoding_type == 'onehot':
            dummies = pd.get_dummies(df_clean[col], prefix=col, dummy_na=False)
            df_clean = pd.concat([df_clean, dummies], axis=1)
            df_clean.drop(columns=[col], inplace=True)
            
        elif encoding_type == 'frequency':
            freq_map = df_clean[col].value_counts(normalize=True).to_dict()
            df_clean[col] = df_clean[col].map(freq_map)
            
        elif encoding_type == 'target' and target_col:
            # Simple Smoothing Target Encoding
            # Global mean
            global_mean = df_clean[target_col].mean()
            # Compute agg
            agg = df_clean.groupby(col)[target_col].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            m = 10 # smoothing weight
            smooth = (counts * means + m * global_mean) / (counts + m)
            df_clean[col] = df_clean[col].map(smooth)
        
        elif encoding_type == 'label':
             le = LabelEncoder()
             df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    return df_clean
