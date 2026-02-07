import time
import torch
import optuna
import main
from Dataset.data import FinetuneDataset
from trainer.Train import Trainer
from model.Model import Model
import pandas as pd

parser = main.parser
args = parser.parse_args()
torch.manual_seed(args.seed)
start_time = time.time()
args.excel_file = 'data/new_bro.xlsx'
dataset = FinetuneDataset(args)


def bayesian_optimization(trial):
    # Suggest parameters
    num_hidden_GCN = trial.suggest_int('num_hidden_GCN', 105, 145)
    num_hidden_MLP1 = trial.suggest_int('num_hidden_MLP1', 175, 198)
    num_hidden_MLP2 = trial.suggest_int('num_hidden_MLP2', 123, 150)

    # Set suggested parameters to args
    args.num_hidden_GCN = num_hidden_GCN
    args.num_layers = 3
    args.num_hidden_MLP1 = num_hidden_MLP1
    args.num_hidden_MLP2 = num_hidden_MLP2

    # Generate a unique model name for this trial
    unique_model_name = f"Model_bayes2_{trial.number}"
    args.model_name = 'bayesian2/' + unique_model_name  # Set the unique model name in args

    # Store the model name in the trial's user attributes
    trial.set_user_attr("model_name", unique_model_name)

    # Create and train the model
    model = Model(args)
    Train = Trainer(model, dataset, args)
    Train.run()

    # Get the validation metrics
    loss = Train.val_loss
    r2 = Train.val_r2
    if not r2:
        return -1.0

    print(f'Model: {unique_model_name}, loss: {loss}, r2: {r2}')
    torch.cuda.empty_cache()
    return r2

def objective(trial):
    return bayesian_optimization(trial)

try:
    # Create an Optuna Study object
    study = optuna.create_study(storage='sqlite:///best2.sqlite3', study_name='best_model_5.7', direction='maximize')
except:
    study = optuna.study.load_study(study_name='best_model_5.7', storage='sqlite:///best2.sqlite3')

# Perform Bayesian optimization search
study.optimize(objective, n_trials=200)

# Retrieve the best trial
trial = study.best_trial

# Get all trial results and save to a DataFrame
results_df = study.trials_dataframe()

# Save trial results to a xlsx file
results_df.to_excel('5.7_results.xlsx', index=False)





