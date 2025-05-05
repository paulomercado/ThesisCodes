import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from scipy.stats import norm, t, gennorm
from sklearn.metrics import mean_absolute_error, mean_squared_error

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows (use wisely for large datasets)
pd.set_option('display.width', 1000)        # Adjust width for better readability
pd.set_option('display.max_colwidth', None) # Prevent truncation of column contents

def generate_resid(
    variable: str,
    train,
    test,
    train_predict,
    test_predict,
    model_results
):
    """
    Generate residuals by dynamically retrieving train and test predictions.
    Ensures train and test are treated as time series.
    """
    
    # Convert train and test to Pandas Series if they are DataFrames
    if isinstance(train, pd.DataFrame):
        train = train.squeeze()
    if isinstance(test, pd.DataFrame):
        test = test.squeeze()
    
    date_range = pd.date_range(start=train.index[0], end=test.index[-1], freq='D')
    
    # Create transformed series
    transformed_series = pd.Series(index=date_range, dtype=float, name=f"{variable.lower()}_complete")
    transformed_series.update(train)
    transformed_series.update(test)
    
    # Create prediction series
    merged_predictions_series = pd.Series(index=date_range, dtype=float, name=f"merged_predictions_series_{variable.lower()}")
    merged_predictions_series.update(train_predict.squeeze() if isinstance(train_predict, pd.DataFrame) else train_predict)
    merged_predictions_series.update(test_predict.squeeze() if isinstance(test_predict, pd.DataFrame) else test_predict)
    
    # Generate train residuals
    resid_train = model_results.resid
    resid_train.name = f"{variable.lower()}_resid_train"
    
    # Compute complete residuals
    resid_complete = transformed_series - merged_predictions_series
    resid_complete.name = f"{variable.lower()}_resid_complete"
    
    # Display results
    display(resid_complete)
    display(resid_train)
    
    return resid_complete, resid_train
    
def garch_testing(residuals, variable: str, max_lag=20):
    """
    Perform ARCH LM test and Portmanteau-Q (Ljung-Box) test for squared residuals,
    and generate diagnostic plots.
    """
    # Plot residuals and squared residuals
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(residuals, label="Residuals", color="blue")
    plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.title(f"SARIMAX {variable} Residuals")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(np.square(residuals), label="Squared Residuals", color="green")
    plt.axhline(0, color="red", linestyle="--", linewidth=0.8)
    plt.title(f"Squared Residuals (SARIMAX {variable})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot PACF of squared residuals
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_pacf(np.square(residuals), lags=max_lag, method="ywm", ax=ax)
    ax.set_title(f"PACF of Squared Residuals ({variable})")
    plt.tight_layout()
    plt.show()
    
    # Perform McLeod-Li Test (Ljung-Box on squared residuals)
    mcleod_li_test = acorr_ljungbox(np.square(residuals), lags=max_lag, return_df=True)
    plt.figure(figsize=(10, 6))
    plt.plot(mcleod_li_test.index, mcleod_li_test['lb_pvalue'], 'o-', label='P-values')
    plt.axhline(0.05, color='red', linestyle='--', linewidth=1, label='Significance Threshold (0.05)')
    plt.title(f"McLeod-Li Test P-Values ({variable})")
    plt.xlabel("Lag")
    plt.ylabel("P-value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Compute ARCH LM and Ljung-Box tests
    lags = list(range(1, max_lag + 1))
    arch_lm_stats, arch_lm_pvalues = [], []
    for lag in lags:
        arch_test = het_arch(residuals, nlags=lag)
        arch_lm_stats.append(arch_test[0])
        arch_lm_pvalues.append(arch_test[1])
    ljungbox_results = acorr_ljungbox(np.square(residuals), lags=lags, return_df=True)
    
    # Store results in DataFrame
    output_df = pd.DataFrame({
        "Lag": lags,
        "PQ Test Stat": ljungbox_results['lb_stat'].values,
        "PQ p-value": ljungbox_results['lb_pvalue'].values,
        "LM Test Stat": arch_lm_stats,
        "LM p-value": arch_lm_pvalues,
    })
    
    return output_df

def evaluate_garch(
    data, 
    vol="GARCH", 
    p_orders=None, 
    q_orders=None, 
    distributions=["normal", "t", "ged"], 
    sort_by="AIC"
):
    """
    Evaluate GARCH/EGARCH models using ACF/PACF insights, AIC/BIC selection, and residual diagnostics.

    Parameters:
        data (array-like): Time series data for fitting the GARCH/EGARCH model.
        vol (str): Type of volatility model to use ("GARCH", "EGARCH", "GJR-GARCH").
        p_orders (list or range): List of p values (lags for ARCH terms) to test. Default is range(0,6).
        q_orders (list or range): List of q values (lags for GARCH terms) to test. Default is range(0,6).
        distributions (list): List of error term distributions to test.
        sort_by (str): Metric to sort results by (options: "AIC", "BIC", "LLF", "ARCH p-value").

    Returns:
        pd.DataFrame: A table showing the AIC, BIC, LLF, residual ARCH test results, alpha/beta parameters, and p-values.
    """
    if not isinstance(data, (pd.Series, np.ndarray)):
        raise ValueError("Input data must be a Pandas Series or a 1D NumPy array.")

    # Convert to Pandas Series and drop NaNs
    data = pd.Series(data).dropna()
    
    # Rescale data if it's too small (for numerical stability)
    if data.abs().mean() < 1e-3:
        data = data * 100  

    # Default orders if none provided
    if p_orders is None:
        p_orders = range(0, 6)  # Default to testing p=0 to p=5
    if q_orders is None:
        q_orders = range(0, 6)  # Default to testing q=0 to q=5

    results = []

    # Iterate through selected (p, q) combinations
    for p in p_orders:
        for q in q_orders:
            if p == 0 and q == 0:
                continue  # Skip (0,0) since it isn't meaningful

            for dist in distributions:
                try:
                    # Fit the GARCH/EGARCH model
                    model = arch_model(data, vol=vol, p=p, q=q, dist=dist, mean="Zero")
                    result = model.fit(disp="off", options={"maxiter": 1000})

                    # Extract key statistics
                    aic = result.aic
                    bic = result.bic
                    llf = result.loglikelihood

                    # Perform ARCH test on residuals
                    arch_test = het_arch(result.resid)
                    arch_stat, arch_pval = arch_test[:2]

                    # Get alpha and beta coefficients with p-values
                    params = result.params
                    pvalues = result.pvalues

                    alpha_params = {f"alpha_{i}": params.get(f"alpha[{i}]", np.nan) for i in range(1, p+1)}
                    alpha_pvals = {f"alpha_pval_{i}": pvalues.get(f"alpha[{i}]", np.nan) for i in range(1, p+1)}

                    beta_params = {f"beta_{i}": params.get(f"beta[{i}]", np.nan) for i in range(1, q+1)}
                    beta_pvals = {f"beta_pval_{i}": pvalues.get(f"beta[{i}]", np.nan) for i in range(1, q+1)}

                    # Store results
                    model_result = {
                        "p": p,
                        "q": q,
                        "Volatility Model": vol,
                        "Distribution": dist,
                        "AIC": aic,
                        "BIC": bic,
                        "LLF": llf,
                        "ARCH Stat": arch_stat,
                        "ARCH p-value": arch_pval,
                    }
                    model_result.update(alpha_params)
                    model_result.update(alpha_pvals)
                    model_result.update(beta_params)
                    model_result.update(beta_pvals)

                    results.append(model_result)

                except Exception as e:
                    # Store NaN results in case of failure
                    model_result = {
                        "p": p,
                        "q": q,
                        "Volatility Model": vol,
                        "Distribution": dist,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "LLF": np.nan,
                        "ARCH Stat": np.nan,
                        "ARCH p-value": np.nan,
                    }
                    results.append(model_result)
                    print(f"Error for {vol}({p},{q}) with {dist} distribution: {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by the specified metric (default is AIC)
    if sort_by in results_df.columns:
        results_df = results_df.sort_values(by=sort_by, ascending=(sort_by not in ["LLF"])).reset_index(drop=True)

    return results_df

def fit_garch(variable: str, vol: str, p: int, q: int, dist: str, last_obs: str, residuals):
    """
    Fit a GARCH(p, q) model to the residuals.
    """
    if residuals is None:
        raise ValueError(f"Residuals for {variable} are not provided.")

    garch_model = arch_model(residuals, vol=vol, p=p, q=q, dist=dist, mean="Zero")
    garch_fitted = garch_model.fit(last_obs=last_obs, disp="off")
    print(garch_fitted.summary())
    return garch_fitted

def garch_predict(
    variable: str, dist: str, model: str, garch_fitted_model,
    train,
    train_predict,
    test_predict
):
    """
    Generate predictions using a SARIMAX-GARCH or SARIMAX-EGARCH model.
    """
    # Set the random seed for reproducibility
    seed = 42
    rs = np.random.RandomState(seed)
    
    # Define a callable for the GARCH forecast's random number generator
    def rng(size):
        return rs.normal(size=size)
    
    # Generate GARCH forecasts
    forecast_horizon = len(test_predict)
    garch_forecast = garch_fitted_model.forecast(
        horizon=forecast_horizon, 
        start=test_predict.index[0], 
        align='origin', 
        reindex=False, 
        method='simulation', 
        simulations=100,
        rng=rng  # Pass the callable for random number generation
    )
    
    forecasted_variance = garch_forecast.variance.iloc[-1, :forecast_horizon].values
    forecasted_stddev = np.sqrt(forecasted_variance)
    
    # Generate random innovations based on the specified distribution
    if dist == "normal":
        simulated_z = rs.normal(loc=0, scale=1, size=forecast_horizon)
    elif dist == "t":
        simulated_z = rs.standard_t(df=garch_fitted_model.params['nu'], size=forecast_horizon)
    elif dist == "ged":
        simulated_z = gennorm.rvs(
            beta=garch_fitted_model.params['nu'], 
            size=forecast_horizon, 
            random_state=rs  # Explicitly pass the RandomState object
        )
    else:
        raise ValueError("Invalid distribution. Choose from 'normal', 't', or 'ged'.")
    
    # Combine predictions
    predicted_et = forecasted_stddev * simulated_z
    predicted_et = pd.Series(predicted_et, index=test_predict.index)
    combined_predictions = test_predict + predicted_et
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    train.plot(ax=ax, label='Train Data', color='blue')
    test_predict.plot(ax=ax, label='SARIMAX Predictions (mu)', color='orange')
    train_predict.plot(ax=ax, label='Train Set Predictions', color='green')
    combined_predictions.plot(ax=ax, label=f'SARIMAX-{model.upper()} Predictions', color='red')
    ax.set_title(f'Predictions with SARIMAX-{model.upper()} Model ({variable.upper()})')
    ax.legend()
    plt.show()
    
    return combined_predictions