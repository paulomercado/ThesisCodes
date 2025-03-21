# Thesis Codes
This repository contains code for our thesis on forecasting electricity prices (GWAP &amp; LWAP) for Luzon, Visayas, and Mindanao. It includes data preprocessing, LSTM, SARIMAX, and GARCH models, plus evaluation methods, incorporating weather, demand, and reserve market data to improve accuracy.
```
## Structure
📂 Final Data/ # Contains processed data files
┣ 📂 raw/ # Contains raw data
┣ 📂 processed/ # Contains aggregated daily data
┗ 📄 *.pkl # Used as input data files for model training
📂 Preds/ # Contains model predictions
┣ 📂 GARCH/ # Predictions from the GARCH/E-GARCH models
┣ 📂 LSTM/ # Predictions from the LSTM model
┗ 📂 sARIMAX/ # Predictions from SARIMAX model
📄 README.md                   # Project documentation
📄 SARIMAX.py                  # SARIMAX model implementation
📄 SARIMAX_LUZ.ipynb           # SARIMAX model for Luzon
📄 SARIMAX_MIN.ipynb           # SARIMAX model for Mindanao
📄 SARIMAX_VIS.ipynb           # SARIMAX model for Visayas
📄 datascript.py               # Script for data preprocessing
📄 model.py                    # LSTM Model-related classes and functions
📄 plot.ipynb                  # Notebook for visualizing results
📄 train.py                    # Training-related functions
📄 transformscript.py          # Script for data transformation
📄 tuning.py                   # Hyperparameter tuning code
📄 utils.py                    # Utility functions
```
