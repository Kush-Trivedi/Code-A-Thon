import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import StackingRegressor, VotingRegressor

class Predictor:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.group_kfold = KFold(n_splits=3)
        self.preprocessor = self.create_preprocessor()
        self.pipelines = {
            'cat_boost': self.create_pipeline(CatBoostRegressor(silent=True)),
            'xg_boost': self.create_pipeline(XGBRegressor()),
            'lgbm': self.create_pipeline(LGBMRegressor(force_col_wise=True,verbosity=-1)),
            'voting': self.create_voting_pipeline(),
            'stacking': self.create_stacking_pipeline()
        }
    
    @staticmethod
    def calculate_metrics(true, predicted, n, k):
        mse = metrics.mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(true, predicted)
        r2_square = metrics.r2_score(true, predicted)
        adjusted_r2 = 1 - (((1 - r2_square) * (n - 1)) / (n - k - 1))

        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Square': r2_square, 'Adjusted R2': adjusted_r2}
    
    @staticmethod
    def evaluate(true, predicted, n, k):  
        metrics_result = Predictor.calculate_metrics(true, predicted, n, k)

        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'MSE', 'RMSE', 'R2 Square', 'Adjusted R2'],
            'Value': [
                round(metrics_result['MAE'], 4),
                round(metrics_result['MSE'], 4),
                round(metrics_result['RMSE'], 4),
                f'{round(metrics_result["R2 Square"] * 100, 4)}%',
                f'{round(metrics_result["Adjusted R2"] * 100, 4)}%'
            ]
        })

        print(metrics_df.to_string(index=False))
        print(20 * '=', '\n')

        return metrics_result
    
    @staticmethod
    def print_metrics(metric_dict):
        metrics_df = pd.DataFrame(metric_dict.items(), columns=['Metric', 'Value'])
        print(metrics_df.to_string(index=False))
        print(20 * '=', '\n')

    def create_preprocessor(self):
        categorical_cols = [cname for cname in self.X.columns if self.X[cname].dtype == "object"]
        numerical_cols = [cname for cname in self.X.columns if self.X[cname].dtype in ['int64', 'float64']]
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('std_scaler', MinMaxScaler())
        ])
        return ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

    def create_pipeline(self, model):
        return Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])

    def create_voting_pipeline(self):
        cat_boost = CatBoostRegressor(silent=True)
        xg_boost = XGBRegressor()
        lgbm = LGBMRegressor(force_col_wise=True,verbosity=-1)
        voting = VotingRegressor(estimators=[('LGBM', lgbm), ('XGB', xg_boost), ('CAT', cat_boost)], weights=[1, 2, 2])
        return self.create_pipeline(voting)

    def create_stacking_pipeline(self):
        lgbm = LGBMRegressor(force_col_wise=True,verbosity=-1)
        xg_boost = XGBRegressor()
        cat_boost = CatBoostRegressor(silent=True)
        layer_one_estimators = [('LGBM', lgbm), ('CAT', cat_boost)]
        layer_two_estimators = [('XGB', xg_boost), ('CAT', cat_boost)]
        layers = StackingRegressor(estimators=layer_two_estimators, final_estimator=xg_boost)
        stacking = StackingRegressor(estimators=layer_one_estimators, final_estimator=layers)
        return self.create_pipeline(stacking)

    def train_evaluate(self, pipeline_key, validation_size=0.3):
        oof_predictions = np.zeros(len(self.X))
        metrics_summary = {'MAE': [], 'MSE': [], 'RMSE': [], 'R2 Square': [], 'Adjusted R2': []}
        
        n = len(self.X)  
        k = len(self.X.columns) 

        for fold, (train_index, test_index) in enumerate(self.group_kfold.split(self.X, self.y)):
            print(20 * '=')
            print(f"Processing Fold {fold + 1}...")
            print(20 * '=', '\n')
          
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

            pipeline = self.pipelines[pipeline_key]
            pipeline.fit(X_train, y_train)


            val_pred = pipeline.predict(X_val)
            print(f"Validation - Fold {fold + 1}")
            val_metrics = self.evaluate(y_val, val_pred, len(X_val), k)
            print("Valid Dataset")
            for key in val_metrics:
                metrics_summary[key].append(val_metrics[key])
            
            test_pred = pipeline.predict(X_test)
            print(f"Test - Fold {fold + 1}")
            test_metrics = self.evaluate(y_test, test_pred, len(X_test), k)
            oof_predictions[test_index] = test_pred
            print("Test Dataset")
            for key in test_metrics:
                metrics_summary[key].append(test_metrics[key])

        oof_metrics = self.calculate_metrics(self.y, oof_predictions, n, k)  
        print(20 * '=')
        print("OOF Metrics:")
        print(20 * '=', '\n')
        self.print_metrics(oof_metrics)

        mean_metrics = {key: np.mean(metrics_summary[key]) for key in metrics_summary}
        print(22 * '=')
        print("Mean Across All Folds:")
        print(22 * '=', '\n')
        self.print_metrics(mean_metrics)
        
    def save_model(self, pipeline_key, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.pipelines[pipeline_key], file)