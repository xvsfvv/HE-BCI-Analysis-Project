import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor

def predict_trends(data, ne_universities):
    """Predict trends for key metrics using multiple methods"""
    # Define metrics to analyze
    metrics = {
        'Research Income': ('research', 'Value'),
        'Business Income': ('business', data['business'].columns[-1]),
        'CPD Income': ('cpd', 'Value'),
        'Regeneration Income': ('regeneration', 'Value'),
        'IP Disclosures': ('ip_disclosures', 'Value'),
        'IP Licenses': ('ip_licenses', 'Value'),
        'IP Income': ('ip_income', 'Value'),
        'Total IP Income': ('ip_income_total', 'Value'),
        'Spin-out Companies': ('spinouts', 'Value'),
        'Public Engagement': ('public_engagement', 'Value')
    }

    methods = [
        ('Linear Regression', 'linear_regression'),
        ('Exponential Smoothing', 'exponential_smoothing'),
        ('Random Forest', 'random_forest')
    ]

    all_methods_summary = {}  # Collect percent change for all methods
    for method_name, method_key in methods:
        results_summary = {}  # Collect all results for this method
        all_methods_summary[method_name] = {}  # Init for this method
        
        for metric_name, (dataset, value_col) in metrics.items():
            results_summary[metric_name] = {}
            all_methods_summary[method_name][metric_name] = {}  # Init for this metric
            
            try:
                df = data[dataset]
                if 'HE Provider' in df.columns and 'Academic Year' in df.columns and value_col in df.columns:
                    df_agg = df.groupby(['HE Provider', 'Academic Year'])[value_col].sum().reset_index()
                else:
                    df_agg = df.copy()
                national_avg = df_agg.groupby('Academic Year')[value_col].mean()
                X_nat, y_nat, years = prepare_time_series(
                    pd.DataFrame({'Academic Year': national_avg.index, 'Value': national_avg.values}),
                    'National Average'
                )
                
                if len(X_nat) > 0:
                    future_years = np.array(range(len(years), len(years) + 3)).reshape(-1, 1)
                    if method_key == 'linear_regression':
                        model_nat = LinearRegression()
                        model_nat.fit(X_nat, y_nat)
                        future_pred_nat = model_nat.predict(future_years)
                    elif method_key == 'exponential_smoothing':
                        try:
                            model_nat = ExponentialSmoothing(y_nat, trend='add', seasonal=None,
                                                             initialization_method="estimated")
                            fit_nat = model_nat.fit()
                            future_pred_nat = fit_nat.forecast(3)
                        except Exception:
                            future_pred_nat = [np.nan] * 3
                    elif method_key == 'random_forest':
                        model_nat = RandomForestRegressor(n_estimators=100, random_state=0)
                        model_nat.fit(X_nat, y_nat)
                        future_pred_nat = model_nat.predict(future_years)
                    
                    results_summary[metric_name]['National Average'] = []
                    for yv, yval in zip(years, y_nat):
                        results_summary[metric_name]['National Average'].append((yv, yval, False))
                    for j, pred in enumerate(future_pred_nat):
                        future_label = f"20{int(years[-1][-2:]) + j + 1}/{int(years[-1][-2:]) + j + 2}"
                        results_summary[metric_name]['National Average'].append((future_label, pred, True))
                
                for university in ne_universities:
                    univ_data = df_agg[df_agg['HE Provider'] == university]
                    if len(univ_data) == 0:
                        continue
                    X, y, _ = prepare_time_series(univ_data, university, value_col)
                    if len(X) > 0:
                        future_years = np.array(range(len(years), len(years) + 3)).reshape(-1, 1)
                        if method_key == 'linear_regression':
                            model = LinearRegression()
                            model.fit(X, y)
                            future_pred = model.predict(future_years)
                        elif method_key == 'exponential_smoothing':
                            try:
                                model = ExponentialSmoothing(y, trend='add', seasonal=None,
                                                             initialization_method="estimated")
                                fit = model.fit()
                                future_pred = fit.forecast(3)
                            except Exception:
                                future_pred = [np.nan] * 3
                        elif method_key == 'random_forest':
                            model = RandomForestRegressor(n_estimators=100, random_state=0)
                            model.fit(X, y)
                            future_pred = model.predict(future_years)
                        
                        results_summary[metric_name][university] = []
                        for yv, yval in zip(univ_data['Academic Year'], y):
                            results_summary[metric_name][university].append((yv, yval, False))
                        for j, pred in enumerate(future_pred):
                            future_label = f"20{int(years[-1][-2:]) + j + 1}/{int(years[-1][-2:]) + j + 2}"
                            results_summary[metric_name][university].append((future_label, pred, True))
                            
            except Exception as e:
                print(f"Error processing {metric_name} with {method_name}: {str(e)}")
                
        # Calculate percentage changes for summary
        for metric, schools in results_summary.items():
            for school, values in schools.items():
                actual_vals = [v for v in values if not v[2]]
                pred_vals = [v for v in values if v[2]]
                if actual_vals and pred_vals:
                    last_actual = actual_vals[-1][1]
                    last_pred = pred_vals[-1][1]
                    if last_actual is not None and last_actual != 0 and not pd.isna(last_actual):
                        pct_change = (last_pred - last_actual) / abs(last_actual) * 100
                        sign = '+' if pct_change >= 0 else ''
                        all_methods_summary[method_name][metric][school] = f"{sign}{pct_change:.2f}%"
                    else:
                        all_methods_summary[method_name][metric][school] = "n/a"
                else:
                    all_methods_summary[method_name][metric][school] = "n/a"
    
    # Generate summary
    md_lines = ["# Trend Prediction Summary\n"]
    for method_name in all_methods_summary:
        md_lines.append(f"## {method_name}\n")
        for metric in all_methods_summary[method_name]:
            md_lines.append(f"### {metric}\n")
            for school in all_methods_summary[method_name][metric]:
                md_lines.append(f"- {school}: {all_methods_summary[method_name][metric][school]}")
            md_lines.append("")
    
    with open("6.trend_prediction.md", "w") as f:
        f.write("\n".join(md_lines))
    
    return all_methods_summary

