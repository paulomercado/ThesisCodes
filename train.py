import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import numpy as np
import os

from pprint import pprint

import transformscript
from utils import set_seed, plot_losses, plot_predictions, Hyperparameter, EarlyStopper
from datascript import TimeSeriesDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, predictions, targets):       
        return torch.sqrt(self.mse(predictions, targets) + 1e-6)

class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon  # Small constant to prevent division by zero

    def forward(self, output, target):
        return torch.mean(torch.abs((target - output) / (target + self.epsilon)))*100
    
L1 = nn.L1Loss()
mape_fn = MAPELoss()


    
def train_model(model, dataloader, device, optimizer, lambda_reg, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)


        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view_as(outputs))

        
        # L1 regularization
        l1_norm = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
        loss += lambda_reg * l1_norm
        
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0

@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0

@torch.no_grad()
def test_best_model(optimized_model, test_loader, test_criterion, regionname, target_label, device):
    optimized_model.eval()

    metrics = {
        'rmse': test_criterion,
        'mae': L1,
        'mape': mape_fn
    }
    
    all_metrics = {metric_name: 0.0 for metric_name in metrics.keys()}
    all_predictions_inverse = []
    all_targets_inverse = []
    total_samples = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)     

        outputs = optimized_model(inputs)
        
        batch_preds = outputs.cpu().numpy()
        batch_targets = targets.cpu().numpy()
        
        # Apply inverse transformation
        batch_preds_inverse = transformscript.inverse_transform_data(batch_preds, 'LSTM',regionname, target_label)
        batch_targets_inverse = transformscript.inverse_transform_data(batch_targets,'LSTM',regionname, target_label)
        
        # Store transformed data
        all_predictions_inverse.append(batch_preds_inverse)
        all_targets_inverse.append(batch_targets_inverse)
        
        # Convert back to tensors for loss calculation
        preds_tensor = torch.tensor(batch_preds_inverse, dtype=torch.float32, device=device)
        targets_tensor = torch.tensor(batch_targets_inverse, dtype=torch.float32, device=device)
        total_samples = total_samples + targets.shape[0]

        for metric_name, metric_fn in metrics.items():
            all_metrics[metric_name] += metric_fn(preds_tensor, targets_tensor).item()*targets.shape[0]

    all_predictions_inverse = np.concatenate(all_predictions_inverse)
    all_targets_inverse = np.concatenate(all_targets_inverse)
    avg_metrics={}

    for metric in metrics.keys():
        avg_metrics[metric] = all_metrics[metric] / total_samples
        print(f"Average {metric} loss: {avg_metrics[metric]:.2f}")

    return avg_metrics,  all_predictions_inverse

def run_hyperparams(config, train_data, val_data, test_data, train_labels, val_labels, test_labels, 
                               input_size, output_size, epoch, island, LSTMModel, 
                               train_criterion, test_criterion, regionname, target_label, output_base_folder):
    num_seeds=10
    set_seed(1)
    hyperparams = Hyperparameter(**config)

    output_folder = os.path.join(output_base_folder, island)
    os.makedirs(output_folder, exist_ok=True)
    
    activation_fn_mapping = {
        "tanh": torch.tanh,
        "relu": F.relu,
        "identity": nn.Identity(),
        "sigmoid": torch.sigmoid
    }
    
    lambda_reg = hyperparams.lambda_reg
    activation_fn = activation_fn_mapping[hyperparams.activation_fn]
    activation_fn1 = activation_fn_mapping[hyperparams.activation_fn1]
    norm_layer_type = hyperparams.norm_layer_type
       
    train_dataset = TimeSeriesDataset(train_data, train_labels, hyperparams.seq_len)
    train_dataloader = DataLoader(train_dataset, hyperparams.batch_size, shuffle=True)

    val_dataset = TimeSeriesDataset(val_data, val_labels, hyperparams.seq_len)    
    val_dataloader = DataLoader(val_dataset, hyperparams.batch_size, shuffle=False)
    
    test_dataset = TimeSeriesDataset(test_data, test_labels, hyperparams.seq_len)    
    test_dataloader = DataLoader(test_dataset, hyperparams.batch_size, shuffle=False)

    metrics = {
        'rmse',
        'mae',
        'mape'
    }
    all_metrics = {metric_name: [] for metric_name in metrics}
    all_pred_inverses = []
    all_train_losses = []
    all_val_losses = []
    
    for seed in range(1, num_seeds + 1):
      
        set_seed(seed)
        optimized_model = LSTMModel(
            input_size=input_size,
            hidden_size=hyperparams.hidden_size,
            output_size=output_size,
            num_layers=hyperparams.num_layers,
            dropout=hyperparams.dropout,
            activation_fn=activation_fn,
            activation_fn1=activation_fn1,
            norm_layer_type=norm_layer_type
        ).to(device)

        optimizer = optim.AdamW(
            optimized_model.parameters(),
            lr=hyperparams.lr,
            weight_decay=hyperparams.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            'min', 
            factor=hyperparams.factor, 
            patience=hyperparams.patience
        )

        train_losses = []
        val_losses = []

        # Train the model with the specified configuration and the current seed
        for e in range(epoch):
            train_loss = train_model(optimized_model, train_dataloader, device, optimizer, lambda_reg, train_criterion)
            val_loss = evaluate(optimized_model, val_dataloader, device, train_criterion)
            scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
 
        
        # Test the model
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        avg_metrics,  pred_inverse = test_best_model(optimized_model, test_dataloader, test_criterion, regionname, target_label,device) 
        for metric in metrics:
            all_metrics[metric].append(avg_metrics[metric])
        all_pred_inverses.append(np.array(pred_inverse))

    # Calculate average loss and predictions
    
    summarize_results = lambda all_metrics: {metric_name: f'{np.mean(values):.2f} ± {np.std(values):.2f}' for metric_name, values in all_metrics.items()}

    pprint(summarize_results(all_metrics))

    stacked_predictions = np.stack(all_pred_inverses)
    avg_pred_inverse = np.mean(stacked_predictions, axis=0)

    for i in range(num_seeds):
        plot_losses(i=i, all_train_losses=all_train_losses, 
                all_val_losses=all_val_losses, 
                island=island, 
                target_label=target_label, 
                output_folder=output_folder)
        
    return all_metrics, avg_pred_inverse