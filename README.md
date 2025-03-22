# Thesis Codes
This repository contains code for our thesis on forecasting electricity prices (GWAP &amp; LWAP) for Luzon, Visayas, and Mindanao. It includes data preprocessing, LSTM, SARIMAX, and GARCH models, plus evaluation methods, incorporating weather, demand, and reserve market data to improve accuracy.

The repository is organized as follows.

```
📂 Final Data/
┣ 📂 Processed/
  ┗ 📄 *.csv
┗ 📄 *.pkl
📂 Graphs/ 
┣ 📂 Luzon/
  ┗ 📄 *.png
┣ 📂 Visayas/
  ┗ 📄 *.png
┗ 📂 Mindanao/
  ┗ 📄 *.png
📂 Preds/ 
┣ 📂 GARCH/
  ┗ 📄 *.csv
┣ 📂 LSTM/
  ┗ 📄 *.csv
┗ 📂 SARIMAX/
  ┗ 📄 *.csv
📄 DM Test.ipynb 
📄 README.md                  
📄 SARIMAX.py                 
📄 SARIMAX_LUZ.ipynb           
📄 SARIMAX_MIN.ipynb           
📄 SARIMAX_VIS.ipynb           
📄 datascript.py
📄 finalconfig.json
📄 main.ipynb                  
📄 model.py                   
📄 plot.ipynb                  
📄 train.py                    
📄 transformscript.py          
📄 tuning.py                   
📄 utils.py                    
```
### Final Data
- `Final Data/` contains input and processed data files.
  -  `Processed/` contains data aggregated into daily values.
  - `*.pkl` files within  `Final Data/` include both input datasets and transformation artifacts required for training

### Graphs
- `Graphs/` contains visualizations related to model performance and results.
  - `Luzon/` contains graphs for Luzon across all models.
  - `Visayas/` contains graphs for Visayas across all models.
  - `Mindanao/` contains graphs for Mindanao across all models.

### Predictions
- `Preds/` contains model predictions.
  - `GARCH/` predictions from the GARCH/E-GARCH models
  - `LSTM/` predictions from the LSTM model.
  - `SARIMAX/` predictions from the SARIMAX model.

### Model and Scripts
- `DM test.ipynb` contains Diebold-Mariano test.
- `SARIMAX.py` contains the SARIMAX model implementation.
- `SARIMAX_LUZ.ipynb` contains the SARIMAX model for Luzon.
- `SARIMAX_MIN.ipynb` contains the SARIMAX model for Mindanao.
- `SARIMAX_VIS.ipynb` contains the SARIMAX model for Visayas.
- `datascript.py` contains the script for data preprocessing.
- `finalconfig.json` contains configuration settings for LSTM model training.
- `main.ipynb` serves as the main notebook for running LSTM experiments.
- `model.py` contains LSTM model-related classes and functions.
- `plot.ipynb` contains the notebook for visualizing results.
- `train.py` contains LSTM training-related functions.
- `transformscript.py` contains the script for data transformation.
- `tuning.py` contains the code for LSTM hyperparameter tuning.
- `utils.py` contains additional utility functions.
