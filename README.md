# Thesis Codes
This repository contains code for our thesis on forecasting electricity prices (GWAP &amp; LWAP) for Luzon, Visayas, and Mindanao. It includes data preprocessing, LSTM, SARIMAX, and GARCH models, plus evaluation methods, incorporating weather, demand, and reserve market data to improve accuracy.
## Structure
```
📂 Final Data/
┣ 📂 raw/ 
┣ 📂 processed/ 
┗ 📄 *.pkl
📂 Preds/ 
┣ 📂 GARCH/
  ┗ 📄 *.csv
┣ 📂 LSTM/
  ┗ 📄 *.csv
┗ 📂 SARIMAX/
  ┗ 📄 *.csv
📄 README.md                  
📄 SARIMAX.py                 
📄 SARIMAX_LUZ.ipynb           
📄 SARIMAX_MIN.ipynb           
📄 SARIMAX_VIS.ipynb           
📄 datascript.py               
📄 model.py                   
📄 plot.ipynb                  
📄 train.py                    
📄 transformscript.py          
📄 tuning.py                   
📄 utils.py                    
```
### Final Data
- `Final Data/` contains input and processed data files.
  - `Raw/` contains raw data files.
  - `Processed/` contains aggregated daily data.
  - `*.pkl` files are used as input data for model training.

### Predictions
- `Preds/` contains model predictions.
  - `GARCH/` predictions from the GARCH/E-GARCH models
  - `LSTM/` predictions from the LSTM model.
  - `SARIMAX/` predictions from the SARIMAX model.

### Model and Scripts
- `SARIMAX.py` contains the SARIMAX model implementation.
- `SARIMAX_LUZ.ipynb` contains the SARIMAX model for Luzon.
- `SARIMAX_MIN.ipynb` contains the SARIMAX model for Mindanao.
- `SARIMAX_VIS.ipynb` contains the SARIMAX model for Visayas.
- `datascript.py` contains the script for data preprocessing.
- `transformscript.py` contains the script for data transformation.
- `train.py` contains training-related functions.
- `tuning.py` contains the code for hyperparameter tuning.
- `model.py` contains LSTM model-related classes and functions.
- `utils.py` contains additional utility functions.
- `plot.ipynb` contains the notebook for visualizing results.
