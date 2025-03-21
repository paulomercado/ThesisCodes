# Import necessary libraries for data manipulation and analysis
import numpy as np # Numerical operations
import pandas as pd # Data manipulation
from datetime import datetime, timedelta # Date operations
from statsmodels.stats.outliers_influence import variance_inflation_factor #Multicollinearity
from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller Test for stationarity check
from statsmodels.tsa.stattools import kpss # Kwiatkowski-Phillips-Schmidt-Shin Test for stationarity check
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # For autocorrelation and partial autocorrelation plots
from statsmodels.tsa.stattools import acf, pacf  # For computing autocorrelation and partial autocorrelation
import matplotlib.pyplot as plt #For visualization
from pmdarima.arima import auto_arima #For order specification
from statsmodels.tsa.statespace.sarimax import SARIMAX # SARIMAX model for time series forecasting
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#Import libraries for transformations
import joblib




#Dealing with multicollinearity in exogenous variables
# Function to calculate VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return (vif_data)

def remove_multicollinear(X, vif_threshold):
     # Check and remove high VIF variables iteratively
    high_vif_threshold = vif_threshold  # Set your threshold here

    while True:
        vif_data = calculate_vif(X)
        max_vif = vif_data['VIF'].max()
        
        if max_vif > high_vif_threshold:
            # Find the variable with the highest VIF
            feature_to_drop = vif_data.loc[vif_data['VIF'] == max_vif, 'feature'].values[0]
            print(f"Dropping {feature_to_drop} with VIF of {max_vif}")
            # Drop the variable and recalculate VIF
            X = X.drop(columns=[feature_to_drop])
        else:
            break
    print("Final VIF values:\n", calculate_vif(X))
    return(X)

#Define a function to remove columns that are multicollinear
def drop_unmatched_columns(df_to_modify, reference_df):
    # Get the columns present in both datasets
    common_columns = df_to_modify.columns.intersection(reference_df.columns)
    # Keep only the columns that are common
    df_modified = df_to_modify[common_columns]
    return df_modified

#Define a function to difference non-stationary data
def perform_differencing(train_data, full_data, max_differences=3):
    # Function to check if data is stationary using ADF
    def adf_test(series):
        result = adfuller(series)
        return result[1]  # p-value

    # Function to check if data is stationary using KPSS
    def kpss_test(series):
        result = kpss(series, regression='c')
        return result[1]  # p-value

    # Check initial stationarity
    for i in range(max_differences):
        adf_p_value = adf_test(train_data)
        kpss_p_value = kpss_test(train_data)

        # If ADF p-value is less than 0.05 and KPSS p-value is greater than 0.05, it's stationary
        if adf_p_value < 0.05 and kpss_p_value > 0.05:
            print(f"Data is stationary after differencing {i} times.")
            return full_data

        # If not stationary, apply differencing
        train_data = train_data.diff().dropna()
        differenced_labels = full_data.diff().dropna()

    print(f"Data could not be made stationary after {max_differences} differencing operations.")
    return differenced_labels

#Define a function to perform differencing iteratively in df
def perform_df_differencing(train_data, full_data, max_differences=3):
    from statsmodels.tsa.stattools import adfuller, kpss

    # Function to check if data is stationary using ADF
    def adf_test(series):
        result = adfuller(series, autolag="AIC")
        return result[1]  # p-value

    # Function to check if data is stationary using KPSS
    def kpss_test(series):
        result = kpss(series, regression="c")
        return result[1]  # p-value

    # Store the differenced datasets in new DataFrames
    diff_train_data = pd.DataFrame(index=train_data.index)
    diff_full_data = pd.DataFrame(index=full_data.index)

    # Loop through each column
    for column in train_data.columns:
        train_series = train_data[column].copy()  # Series for train_data
        full_series = full_data[column].copy()  # Series for full_data

        for i in range(max_differences):
            adf_p_value = adf_test(train_series)
            kpss_p_value = kpss_test(train_series)

            # If stationary, stop differencing and store the results
            if adf_p_value < 0.05 and kpss_p_value > 0.05:
                print(f"Column '{column}' is stationary after differencing {i} times.")
                diff_train_data[column] = train_series
                diff_full_data[column] = full_series
                break

            # Apply differencing if not stationary
            if i < max_differences - 1:  # Don't drop NA on the last iteration
                train_series = train_series.diff().dropna()
                full_series = full_series.diff().dropna()

        else:  # If the loop completes without breaking
            print(f"Column '{column}' could not be made stationary after {max_differences} differencing operations.")
            diff_train_data[column] = train_series
            diff_full_data[column] = full_series

    return diff_full_data

#Plotting
def plot_orders(df, lags=25, alpha=0.05, var=''):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    print(f"The ACF for {var} is seen below")
    plot_acf(df, lags=lags, alpha=alpha, ax=axs[0])

    print(f"The PACF for {var} is seen below")
    plot_pacf(df, lags=lags, alpha=alpha, ax=axs[1])

    plt.tight_layout()
    plt.show()

#Define a function to  run auto-arima on seasonalities 0-7.14,30
def seasonal_auto_arima(df, max_p, max_d, max_q, exog_data=None):
    lowest_aic = float("inf")
    best_model = None
    
    # Sequence of seasonal periods to try
    seq = [1,2,3,4,5,6,7,14,30]
    for i in seq:
        try:
            model = auto_arima(df, 
                               start_p=0, start_d=0, start_q=0,
                               max_p=max_p, max_d=max_d, max_q=max_q, 
                               seasonal=True, m=i, 
                               start_P=0, start_D=0, start_Q=0,
                               error_action='warn', trace=False,
                               suppress_warnings=True, stepwise=False,
                               exog=exog_data, 
                               maxiter=300, method='powell')
            
            if model.aic() < lowest_aic:
                lowest_aic = model.aic()
                best_model = model

        except Exception as e:
            print(f"Model fitting failed for seasonality {i}: {e}")

    try:
        model = auto_arima(df, 
                           start_p=0, start_d=0, start_q=0,
                           max_p=max_p, max_d=max_d, max_q=max_q, 
                           seasonal=False,
                           error_action='warn', trace=False,
                           suppress_warnings=True, stepwise=False,
                           exog=exog_data, 
                           maxiter=300, method='powell')
        
        if model.aic() < lowest_aic:
            lowest_aic = model.aic()
            best_model = model

    except Exception as e:
        print(f"Non-seasonal model fitting failed: {e}")

    if best_model is not None:
        if best_model.seasonal_order == (0, 0, 0, 0):
            print(f"The model with the lowest AIC is non-seasonal with an AIC of {lowest_aic}")
        else:
            print(f"The model with the lowest AIC has a seasonal period of {best_model.seasonal_order[3]} and an AIC of {lowest_aic}")
    else:
        print("No suitable model found.")
        
    return best_model

#Define a function to fit the SARIMAX models for GWAP and LWAP, check fits, and predict on test set
def fit_SARIMAX(gwap_endog, lwap_endog, train_exog, gwap_order, gwap_seasonal_order, GWAP_test, lwap_order, lwap_seasonal_order, LWAP_test, exog_test):
    # Fit GWAP SARIMAX model
    gwap_model = SARIMAX(endog=gwap_endog, exog=train_exog, order=gwap_order, seasonal_order=gwap_seasonal_order)
    gwap_model_results = gwap_model.fit(method='powell', maxiter=300)

    # Predict on train set
    gwap_train_predict = gwap_model_results.predict(start=gwap_endog.index[0], end=gwap_endog.index[-1], exog=train_exog)
    fig, ax = plt.subplots(figsize=(7, 3))
    gwap_endog.plot(ax=ax, label='train')
    gwap_train_predict.plot(ax=ax, label='predicted')
    ax.set_title('Predictions with GWAP SARIMAX model')
    ax.legend()

    # Display summary of results
    print(gwap_model_results.summary())

    # Conduct residual analysis by plotting residuals
    gwap_model_results.plot_diagnostics(figsize=(9, 9))

    # Fit LWAP SARIMAX model
    lwap_model = SARIMAX(endog=lwap_endog, exog=train_exog, order=lwap_order, seasonal_order=lwap_seasonal_order)
    lwap_model_results = lwap_model.fit(method='powell', maxiter=300)

    # Predict on train set
    lwap_train_predict = lwap_model_results.predict(start=lwap_endog.index[0], end=lwap_endog.index[-1], exog=train_exog)
    fig, ax = plt.subplots(figsize=(7, 3))
    lwap_endog.plot(ax=ax, label='train')
    lwap_train_predict.plot(ax=ax, label='predicted')
    ax.set_title('Predictions with LWAP SARIMAX model')
    ax.legend()

    # Display summary of results
    print(lwap_model_results.summary())

    # Conduct residual analysis by plotting residuals
    lwap_model_results.plot_diagnostics(figsize=(9, 9))

    #Append model results to original data
    #This will allow the model to predict values for time t+1 by using actual data from time 1 to t
    appended_gwap = gwap_model_results.append(GWAP_test, exog=exog_test, refit=False)
    appended_lwap = lwap_model_results.append(LWAP_test, exog=exog_test, refit=False)

    #Forecast day-ahead
    gwap_test_predict = appended_gwap.predict(start=GWAP_test.index[0], end=GWAP_test.index[-1], exog=exog_test, dynamic=False)
    lwap_test_predict = appended_lwap.predict(start=LWAP_test.index[0], end=LWAP_test.index[-1], exog=exog_test, dynamic=False)

    #Visualize SARIMAX predictions on test date
    fig, ax = plt.subplots(figsize=(7, 3))
    gwap_endog.plot(ax=ax, label='train')
    gwap_train_predict.plot(ax=ax, label='Predictions on Train Set')
    gwap_test_predict.plot(ax=ax, label='GWAP Predictions')
    ax.set_title('GWAP Predictions with SARIMAX models')
    ax.legend()

    fig, ax = plt.subplots(figsize=(7, 3))
    lwap_endog.plot(ax=ax, label='train')
    lwap_train_predict.plot(ax=ax, label='Predictions on Train Set')
    lwap_test_predict.plot(ax=ax, label='LWAP Predictions')
    ax.set_title('LWAP Predictions with SARIMAX models')
    ax.legend()


    return gwap_test_predict, lwap_test_predict, gwap_train_predict, lwap_train_predict

#Define a function to calculate MAE and RMSE
def evaluate_models(GWAP_original, LWAP_original, GWAP_test_original, GWAP_predictions_inverse, LWAP_test_original, LWAP_predictions_inverse, Region):
    # Calculate MAE for the GWAP test set
    gwap_mae_test = mean_absolute_error(GWAP_test_original, GWAP_predictions_inverse)
    print(f"Mean Absolute Error (MAE) on Test Set for GWAP: {gwap_mae_test}")

    # Calculate MSE on the test set
    gwap_mse_test = mean_squared_error(GWAP_test_original, GWAP_predictions_inverse)

    # Calculate RMSE on the test set
    gwap_rmse_test = np.sqrt(gwap_mse_test)
    print(f"Root Mean Squared Error (RMSE) on Test Set for GWAP: {gwap_rmse_test}")

    # Calculate MAPE for the GWAP test set
    mape_test_gwap = np.mean(np.abs((GWAP_test_original - GWAP_predictions_inverse) / GWAP_test_original)) * 100
    print(f"Mean Absolute Percentage Error (MAPE) on Test Set for GWAP: {mape_test_gwap}")

    # Create a new figure and Axes object
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    GWAP_original.squeeze().plot(ax=ax1, label="Original GWAP")  # Convert to Series
    GWAP_predictions_inverse.squeeze().plot(ax=ax1, linestyle='--', label='GWAP Predictions')  # Convert to Series
    ax1.set_title(f'SARIMAX GWAP {Region} Predictions')
    ax1.legend()
    plt.show()

    # Repeat for LWAP
    # Calculate MAE for the LWAP test set
    lwap_mae_test = mean_absolute_error(LWAP_test_original, LWAP_predictions_inverse)
    print(f"Mean Absolute Error (MAE) on Test Set for LWAP: {lwap_mae_test}")

    # Calculate MSE on the test set
    lwap_mse_test = mean_squared_error(LWAP_test_original, LWAP_predictions_inverse)

    # Calculate RMSE for the LWAP test set
    lwap_rmse_test = np.sqrt(lwap_mse_test)
    print(f"Root Mean Squared Error (RMSE) on Test Set for LWAP: {lwap_rmse_test}")

    # Calculate MAPE for the LWAP test set
    mape_test_lwap = np.mean(np.abs((LWAP_test_original - LWAP_predictions_inverse) / LWAP_test_original)) * 100
    print(f"Mean Absolute Percentage Error (MAPE) on Test Set for LWAP: {mape_test_lwap}")

    # Plot predictions
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    LWAP_original.squeeze().plot(ax=ax2, label='Original Data')  # Convert to Series
    LWAP_predictions_inverse.squeeze().plot(ax=ax2, label='Predictions LWAP', linestyle='--')  # Convert to Series
    ax2.set_title(f'SARIMAX {Region} LWAP Predictions')
    ax2.legend()
    plt.show()

    return gwap_mae_test, gwap_rmse_test, mape_test_gwap, mape_test_lwap, gwap_rmse_test, lwap_rmse_test


