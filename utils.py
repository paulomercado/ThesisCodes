import numpy as np
import pandas as pd

import torch
import random
import os

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dieboldmariano import dm_test

class Hyperparameter:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
            
def set_seed(seed=None):
    if seed is None:
        seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

def conduct_dm(actual, lstm, time_series, region, target_variable, statmodel):
    # Perform the Diebold-Mariano test
    dm, pvalue = dm_test(actual, lstm, time_series)
    
    # Print the test results
    if pvalue < 0.05:
        print(f"There is a significant difference between LSTM and {statmodel} in predicting {region} {target_variable}.")
    else:
        print(f"LSTM has the same predictive accuracy as {statmodel} in predicting {region} {target_variable}.")
    
    print(f"The DM Test Statistic is {dm[0]:.4f}")  # Display the statistic with 4 decimal places
    print(f"The p-value is {pvalue[0]:.4f}")       # Display the p-value with 4 decimal places

font = {'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 15,
        }

def plot_predictions(actual_wap, target_label, island, output_folder,appendix,**kwargs):
    if island == "Mindanao":
        full_date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D").to_numpy()
    else:
        full_date_range = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D").to_numpy()
    colors = ['red','green', 'blue',  'purple']
    test_data_points = next(iter(kwargs.values())).shape[0]
    test_start_date = datetime(2023, 12, 31) - timedelta(days=test_data_points - 1)
    test_date_range = [test_start_date + timedelta(days=i) for i in range(test_data_points)]
    
    plt.figure(figsize=(12, 6))
    if appendix:
        plt.plot(full_date_range, actual_wap.values, label=f'{target_label} Actual (Full Data)', color='k')
        for i, (model_name, preds) in enumerate(kwargs.items()):
            plt.plot(test_date_range, preds[-test_data_points:], label=f'{target_label} Prediction ({model_name})',
                    color=colors[i % len(colors)],linewidth=1.1,alpha=0.7)
        plt.xticks(rotation=45)
    else:
        plt.plot(test_date_range, actual_wap[-test_data_points:].values, label=f'{target_label} Actual (Full Data)', color='k')
      
        for i, (model_name, preds) in enumerate(kwargs.items()):
            plt.plot(test_date_range, preds[-test_data_points:], label=f'{target_label} Prediction ({model_name})',
                    color=colors[i % len(colors)],linewidth=1.1,alpha=0.7)

        monthly_ticks = []
        monthly_labels = []
        
        monthly_ticks.append(test_date_range[0])
        monthly_labels.append(test_date_range[0].strftime('%Y-%m-%d'))
        
        # Add monthly ticks
        current_month = test_start_date.month
        for i, date in enumerate(test_date_range):
            # If this is the first day of a new month, add it as a tick
            if date.month != current_month:
                monthly_ticks.append(date)
                monthly_labels.append(date.strftime('%Y-%m-%d'))
                current_month = date.month
        
        # Add January 2024 (outside the dataset)
        jan2024 = datetime(2024, 1, 1)
        
        # Extend the x-axis slightly to include Jan 2024
        plt.xlim(test_date_range[0], jan2024)
        monthly_ticks.append(jan2024)
        monthly_labels.append(jan2024.strftime('%Y-%m-%d'))
        
        plt.xticks(monthly_ticks, monthly_labels, rotation=45)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xlabel('Date', fontdict=font)
    plt.ylabel('PHP/MWh', fontdict=font)
    plt.tight_layout()  # Improve layout to prevent cut-off labels
    
    plt.legend(loc='upper left')
    
    os.makedirs(output_folder, exist_ok=True)
    if appendix:
        plt.savefig(os.path.join(output_folder, "Appendix",f'{island}_{target_label}_preds.png'))
    else:
        plt.savefig(os.path.join(output_folder, f'{island}_{target_label}_preds.png'))
    plt.show()


def plot_losses(i,all_train_losses,all_val_losses,island,target_label,output_folder):
    
    plt.figure(figsize=(12, 6))

    plt.plot(all_train_losses[i], label=f'Training Loss Seed {i+1}')
    plt.plot(all_val_losses[i], label=f'Validation Loss Seed {i+1}')

    plt.grid(True)
    
    plt.xlabel('Epoch',fontdict=font)
    plt.ylabel('Loss',fontdict=font)
    plt.xticks(rotation=45)
    plt.ylim(0, 0.5)
    
    plt.legend(loc='upper left')
    plt.title(f"{island} {target_label} Training and Validation Losses per Epoch for Seed {i+1}",fontdict=font)
    
    # Save the figure
    output_path = os.path.join(output_folder, f"{island}_{target_label}_training_validation_loss_seed_{i+1}.png")
    plt.savefig(output_path)
    plt.show()