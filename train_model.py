import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from itertools import product
from sklearn.model_selection import train_test_split, GridSearchCV
from pandas.api.types import CategoricalDtype
from data_pipeline import DataPipeline
from pathlib import Path

class TrainModel:

    def __init__(self, data_path, artifacts_path = "../artifacts/revenue_model.pkl"):
    # ==========================================
    # 1. Configuration
    # ==========================================
    ## NOTE: data_path should be an excel file
        self.data_path = data_path
        self.artifacts_path = artifacts_path
        if artifacts_path != "../artifacts/revenue_model.pkl":
            print("artifacts_path changed, take note!")

    def sync_categorical_types(self, X_train, X_val):
        """
        Creates a unified CategoricalDtype based on ALL known categories in Train/Val.
        Returns a dictionary of {col_name: CategoricalDtype} to save for production.
        """
        cat_cols = X_train.select_dtypes(include=['category', 'object']).columns
        cat_schema = {}

        for col in cat_cols:
            # Get union of categories from Train and Val
            # We assume X_train and X_val have already run through the DataPipeline
            known_cats = set(X_train[col].dropna().unique()) | set(X_val[col].dropna().unique())
            
            # Create strict type
            cat_type = CategoricalDtype(categories=list(known_cats), ordered=False)
            cat_schema[col] = cat_type
            
            # Apply immediately
            X_train[col] = X_train[col].astype(cat_type)
            X_val[col] = X_val[col].astype(cat_type)

            # xgboost requires input as numeircal even though it has to be categorical
            X_train[col] = X_train[col].cat.codes
            X_val[col] = X_val[col].cat.codes

            # force dtype to be "category" - to solve the "ordinal" problem
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")
            
        return X_train, X_val, cat_schema

    def tune_parameters(self, base_model, param_grid, X_t, y_t, X_v, y_v):
        """Helper to run parameter tuning using native XGBoost API instead of GridSearchCV
        to avoid categorical type conflicts with sklearn's internal splitting"""
        print(f"  > Tuning: {list(param_grid.keys())}...")
        
        # Keep categorical dtypes - XGBoost DMatrix handles them with enable_categorical
        dtrain = xgb.DMatrix(X_t, label=y_t, enable_categorical=True)
        dval = xgb.DMatrix(X_v, label=y_v, enable_categorical=True)
        
        best_score = float('inf')
        best_params = {}
        
        # Manual grid search using native XGBoost
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Get current params from base_model to use as defaults
        current_model_params = base_model.get_params()
        base_xgb_params = {
            'objective': current_model_params.get('objective', 'reg:squarederror'),
            'learning_rate': current_model_params.get('learning_rate', 0.05),
            'subsample': current_model_params.get('subsample', 0.8),
            'colsample_bytree': current_model_params.get('colsample_bytree', 0.8),
            'n_estimators': current_model_params.get('n_estimators', 1000),
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': current_model_params.get('random_state', 42),
            'max_depth': current_model_params.get('max_depth', 6),
            'min_child_weight': current_model_params.get('min_child_weight', 5),
            'reg_alpha': current_model_params.get('reg_alpha', 0),
            'reg_lambda': current_model_params.get('reg_lambda', 1.0),
        }
        
        for values in product(*param_values):
            current_params = dict(zip(param_names, values))
            # Merge with base params (current_params override)
            full_params = {**base_xgb_params, **current_params}
            
            # Train with early stopping
            evals = [(dtrain, 'train'), (dval, 'eval')]
            evals_result = {}
            
            bst = xgb.train(
                full_params,
                dtrain,
                num_boost_round=full_params['n_estimators'],
                evals=evals,
                evals_result=evals_result,
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Get best eval score
            final_eval_score = evals_result['eval']['rmse'][-1]
            print(f"    - Params {current_params}: RMSE = {final_eval_score:.4f}")
            
            if final_eval_score < best_score:
                best_score = final_eval_score
                best_params = current_params
        
        print(f"    - Best Params: {best_params}")
        print(f"    - Best Score (RMSE): {best_score:.4f}")
        
        # Update the XGBRegressor with best params
        base_model.set_params(**best_params)
        return base_model

    def run(self):
        # 1. Data Loading & Splitting (Must happen BEFORE Pipeline to avoid leakage)
        print(">> Loading and Splitting Raw Data...")
        df_raw = pd.read_excel(self.data_path)
        
        # Split Raw Data
        X = df_raw.drop('Revenue (USD)', axis=1) # Assuming this is target
        # y = np.log1p(df_raw['Revenue (USD)']) # Log Transform Target immediately
        y = df_raw['Revenue (USD)']

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Run Data Pipeline (Training Mode)
        # We instantiate the pipeline passing the DATAFRAMES directly (requires slight tweak to your class)
        # OR simpler: Save temp CSVs and let your class load them (easier integration with your existing code)
        print(">> Running Data Pipeline...")
        # NOTE: Assuming you adapted DataPipeline to accept a dataframe or path. 
        # For now, let's assume we initialize it and inject the split data.
        
        # Initialize Pipeline (Training Mode)
        pipeline = DataPipeline(df = pd.concat([X_train_raw, y_train], axis = 1), 
                                is_training=True)
        df_train_clean, train_artifact_info = pipeline.run() # This saves encodings to disk
        
        # Now process Validation set (Prediction Mode - reuse encodings)
        pipeline_val = DataPipeline(df = pd.concat([X_val_raw, y_val], axis = 1),
                                    is_training=False)
        # Ensure artifacts are loaded or shared
        pipeline_val.load_encoding_artifacts() # load saved artifacts
        df_val_clean, _ = pipeline_val.run()

        # Separating X and y after cleaning
        X_train_xg = df_train_clean.drop('Log_Revenue (USD)', axis=1) # Adjust col name based on your pipeline output
        y_train = df_train_clean['Log_Revenue (USD)']
        
        X_val_xg = df_val_clean.drop('Log_Revenue (USD)', axis=1)
        y_val = df_val_clean['Log_Revenue (USD)']

        # 3. Synchronize Categories
        print(">> Synchronizing Categories...")
        X_train_xg, X_val_xg, cat_schema = self.sync_categorical_types(X_train_xg, X_val_xg)

        # 4. Initialize Base Model
        xgb_reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=10000,
            learning_rate=0.015,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=True,
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=100,
            eval_metric="rmse",
            min_child_weight=5
        )

        # 5. Hyperparameter Tuning Phases
        print(">> Phase 2: Tuning Structure...")
        param_structure = {'max_depth': [4, 6, 8], 'min_child_weight': [10, 30]}
        xgb_reg =self.tune_parameters(xgb_reg, param_structure, X_train_xg, y_train, X_val_xg, y_val)

        print(">> Phase 3: Tuning Randomness...")
        param_noise = {'subsample': [0.5, 0.7, 0.9], 'colsample_bytree': [0.7, 0.9, 1.0]}
        xgb_reg = self.tune_parameters(xgb_reg, param_noise, X_train_xg, y_train, X_val_xg, y_val)

        print(">> Phase 4: Tuning Regularization...")
        param_reg = {'reg_alpha': [0, 0.1, 1.0], 'reg_lambda': [1.0, 5.0, 7.0]}
        xgb_reg = self.tune_parameters(xgb_reg, param_reg, X_train_xg, y_train, X_val_xg, y_val)

        # 6. Final Production Fit
        print(">> Phase 5: Production Training (Full Fit)...")
        xgb_reg.set_params(learning_rate=0.0075, n_estimators=10000)
        
        # Keep categorical dtypes - XGBoost will handle them natively with enable_categorical=True
        xgb_reg.fit(
            X_train_xg, y_train,
            eval_set=[(X_train_xg, y_train), (X_val_xg, y_val)],
            verbose=100
        )

        # 7. SAVE EVERYTHING
        print(f">> Saving Model Artifacts to {self.artifacts_path}...")
        
        # Create artifacts directory if it doesn't exist
        Path(self.artifacts_path).parent.mkdir(parents=True, exist_ok=True)
        
        bundle = {
            "model": xgb_reg,
            "cat_schema": cat_schema, # Crucial for prediction
            "features": list(X_train_xg.columns), # Safety check
            "pipeline_metadata": {
                "score": xgb_reg.best_score,
                "best_iteration": xgb_reg.best_iteration
            }
        }
        
        joblib.dump(bundle, self.artifacts_path)
        print(">> Training Complete. Ready for Production.")

# # UNCOMMENT TO RUN
train = pd.read_excel("../data/train_df.xlsx")
val = pd.read_excel("../data/val_df.xlsx")
test = pd.read_excel("../data/test_df.xlsx")
df = pd.concat([train, test], axis = 0)
df.to_excel("../data/danny_final_test.xlsx", index = False)
xgboost_model = TrainModel(data_path = "../data/danny_final_test.xlsx")
xgboost_model.run()