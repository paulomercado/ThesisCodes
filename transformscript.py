import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
import pickle
from datascript import split_data
import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classify_features(data):
    minmax_cols, boxcox_cols, yeojohnson_cols = [], [], []
    for column in data.columns:
        skewness = data[column].skew()
        kurtosis = data[column].kurtosis()
        is_positive = np.all(data[column] > 0)

        if -1 <= skewness <= 1 and -1 <= kurtosis <= 1:
            minmax_cols.append(column)
        elif is_positive:
            boxcox_cols.append(column)
        else:
            yeojohnson_cols.append(column)
    return minmax_cols, boxcox_cols, yeojohnson_cols

def fit_pipeline(df, cols, model,regionname,target_label,method='box-cox'):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    pickle_file = None
    if len(cols)==1:
        pickle_file = os.path.join(script_dir, "Final Data", f"{model}_{regionname}_target_{target_label}_{method}_pipeline.pkl")

    if pickle_file and os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    
    if method == "box-cox" or method == "yeo-johnson":
        pipeline = Pipeline([
            ('power_transformer', PowerTransformer(method=method, standardize=False)),
            ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
    elif method == "minmax":
        pipeline = Pipeline([
            ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
        ])

    pipeline.fit(df[cols])
    if pickle_file:  
        with open(pickle_file, 'wb') as f:
            pickle.dump(pipeline, f)
    return pipeline

def transform_data(data, model, regionname, target_label, use_val=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    feature_pickle = os.path.join(script_dir, "Final Data", f"{model}_{regionname}_{target_label}_Transformed_Features.pkl")
    target_pickle = os.path.join(script_dir, "Final Data", f"{model}_{regionname}_{target_label}_Transformed_Target.pkl")

    # Load cached transformations if available
    if os.path.exists(feature_pickle) and os.path.exists(target_pickle):
        with open(feature_pickle, "rb") as f:
            transformed_features = pickle.load(f)
        with open(target_pickle, "rb") as f:
            transformed_labels = pickle.load(f)
        return transformed_features, transformed_labels

    # Ensure target exists
    if target_label not in data.columns:
        raise ValueError(f"Target '{target_label}' not found in dataset.")

    # Sort and set index if needed
    features = data.drop(columns=[col for col in ["GWAP", "LWAP"] if col != target_label])
    features = features.sort_values(by="RUN_TIME").set_index("RUN_TIME") if "RUN_TIME" in features.columns else features.sort_index()
    
    # Feature classification
    minmax_cols, boxcox_cols, yeojohnson_cols = classify_features(features)

    # Split data
    if use_val:
        train, val, test = split_data(features, True)
        train_labels, val_labels, test_labels = train[[target_label]], val[[target_label]], test[[target_label]]
    else:
        train, test = split_data(features, False)
        train_labels, test_labels = train[[target_label]], test[[target_label]]

    # Fit transformation pipelines
    pipelines = {
        "boxcox": fit_pipeline(train[boxcox_cols], boxcox_cols, model, regionname, target_label, "box-cox"),
        "yeojohnson": fit_pipeline(train[yeojohnson_cols], yeojohnson_cols, model, regionname, target_label, "yeo-johnson"),
        "minmax": fit_pipeline(train[minmax_cols], minmax_cols, model, regionname, target_label, "minmax")
    }
    label_pipeline = fit_pipeline(train_labels, [target_label], model, regionname, target_label, "box-cox")

    # Apply transformations
    def apply_transform(df):
        return pd.concat(
            [pd.DataFrame(pipelines[method].transform(df[cols]), columns=cols, index=df.index)
             for method, cols in [("minmax", minmax_cols), ("boxcox", boxcox_cols), ("yeojohnson", yeojohnson_cols)] if cols],
            axis=1
        )[df.columns]

    transformed_features = pd.concat([apply_transform(df) for df in ([train, val, test] if use_val else [train, test])])
    transformed_labels = pd.concat([
        pd.DataFrame(label_pipeline.transform(df), index=df.index, columns=[target_label])
        for df in ([train_labels, val_labels, test_labels] if use_val else [train_labels, test_labels])
    ])

    # Save and return
    with open(feature_pickle, "wb") as f:
        pickle.dump(transformed_features, f)
    with open(target_pickle, "wb") as f:
        pickle.dump(transformed_labels, f)

    return transformed_features, transformed_labels

def inverse_transform_data(data, model, regionname, target_label):
    script_dir = os.path.dirname(os.path.abspath(__file__))
   
    # Check if data is numpy array or tensor and convert to DataFrame
    original_type = None
    original_shape = None
    pipeline_file = None

    if isinstance(data, np.ndarray):
        original_type = "numpy"
        original_shape = data.shape
        
        data = pd.DataFrame(data.reshape(-1, 1), columns=[target_label])
        
    elif isinstance(data, torch.Tensor):
        original_type = "tensor"
        data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
        original_shape = data_np.shape
        
        data = pd.DataFrame(data_np.reshape(-1, 1), columns=[target_label])
    
    if len(data.columns) == 1:
        pipeline_file = os.path.join(script_dir, "Final Data", f"{model}_{regionname}_target_{target_label}_box-cox_pipeline.pkl")
        
    if pipeline_file and os.path.exists(pipeline_file):
        with open(pipeline_file, "rb") as f:
            pipeline = pickle.load(f)
    else:
        raise FileNotFoundError(f"Pipeline file {pipeline_file} not found. Ensure transformations were applied first.")

    boxcox_cols = pipeline.named_steps["power_transformer"].feature_names_in_

    result_df = pd.DataFrame(
        pipeline.inverse_transform(data[boxcox_cols]), 
        columns=boxcox_cols, 
        index=data.index
    )
    
    # Return the result in the same format as the input
    if original_type == "numpy":
        result_np = result_df.values
        # Reshape back to original shape if needed (handles 1D case)
        if len(original_shape) == 1:
            result_np = result_np.squeeze()
        return result_np
    elif original_type == "tensor":
        result_np = result_df.values
        if len(original_shape) == 1:
            result_np = result_np.squeeze()
        return torch.tensor(result_np)
    else:
        return result_df


