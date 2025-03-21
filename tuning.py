import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from ray.tune.search.basic_variant import BasicVariantGenerator

from ray import tune, train
from ray.tune import CLIReporter, Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import TrialPlateauStopper
from ray.train import RunConfig

from datascript import TimeSeriesDataset
from train import train_model, evaluate
from utils import Hyperparameter,  set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

activation_fn_mapping = {
        "tanh": torch.tanh,
        "relu": F.relu,
        "identity": nn.Identity(),
        "sigmoid": torch.sigmoid
    }

def raytrain(config, epoch, train_data, val_data, train_labels, val_labels, input_size, output_size, train_criterion, LSTMModel):

    set_seed(1)
    hyperparams = Hyperparameter(**config)

    lambda_reg = hyperparams.lambda_reg
    norm_layer_type = hyperparams.norm_layer_type
    activation_fn = activation_fn_mapping[hyperparams.activation_fn]
    activation_fn1 = activation_fn_mapping[hyperparams.activation_fn1]
    
    model = LSTMModel(
        input_size, 
        hyperparams.hidden_size, 
        output_size, 
        hyperparams.num_layers,
        hyperparams.dropout,
        activation_fn=activation_fn,
        activation_fn1=activation_fn1, 
        norm_layer_type=norm_layer_type
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor= hyperparams.factor, patience=hyperparams.patience)
    
    train_dataset = TimeSeriesDataset(train_data, train_labels, hyperparams.seq_len)
    train_dataloader = DataLoader(train_dataset, hyperparams.batch_size, shuffle=True)

    val_dataset = TimeSeriesDataset(val_data, val_labels, hyperparams.seq_len)    
    val_dataloader = DataLoader(val_dataset, hyperparams.batch_size, shuffle=False) 
    
    for e in range(epoch):
        train_model(model, train_dataloader, device, optimizer, lambda_reg, train_criterion)
        val_loss = evaluate(model, val_dataloader, device, train_criterion)
        scheduler.step(val_loss)
        train.report({"loss": val_loss})
    

def tunemodel(train_data, val_data, train_labels, val_labels,  
                   input_size, output_size, epoch, trials, LSTMModel, 
                   train_criterion,initialconfig):
    set_seed(1)
    
    
    search_space = {
        "hidden_size": tune.choice([16, 32,50, 64, 100,128,150,200, 256, 512]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "dropout": tune.uniform(0.0, 0.5),
        "activation_fn": tune.choice(["tanh", "relu"]),
        "activation_fn1": tune.choice(["sigmoid", "identity"]),
        "seed": tune.choice([1]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "patience": tune.choice([5, 10, 20,50,100]),
        "factor": tune.uniform(0.1, 0.5),
        "seq_len": tune.choice([3,5,7, 9,  11,  13,  15]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "norm_layer_type": tune.choice(['batch_norm', 'layer_norm', 'none']),
        "lambda_reg": tune.loguniform(1e-5, 1e-1)
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epoch,
        grace_period=50,
        reduction_factor=2
    )
    
    search_alg = OptunaSearch(points_to_evaluate=initialconfig, metric="loss", mode="min")
    plateau_stopper = TrialPlateauStopper(metric="val_loss", mode="min")

    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"]
    )

    def trial_dirname_creator(trial):
        return f"trial_{trial.trial_id}"

    trainable_with_params = tune.with_parameters(
        raytrain, 
        epoch=epoch,
        train_data=train_data,
        val_data=val_data,
        train_labels=train_labels,
        val_labels=val_labels,
        input_size=input_size,
        output_size=output_size,
        train_criterion=train_criterion,
        LSTMModel=LSTMModel
    )

    trainable_with_resources = tune.with_resources(
        trainable_with_params, 
        resources={"cpu": 5, "gpu": 0.25,'accelerator_type:G':0.25}
    )

    tuner = Tuner(
        trainable_with_resources,  
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            num_samples=trials,
            scheduler=scheduler,
            trial_dirname_creator=trial_dirname_creator
        ),
        run_config=RunConfig(
            progress_reporter=reporter,
            verbose=1,
            stop=plateau_stopper
        )
    )
    
    # Run the tuner and collect the results
    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")

    return best_result.config