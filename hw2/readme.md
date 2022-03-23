# Report

## Method

* BiLSTM + Cosine Anealing LR scheduler.
* Use `optuna` package to find best combinations of hyperparameters.
* Vote result of four different combinations of hyperparameters.

## Run Code

### Prepare Requirements

* pip install -r requirements.txt

### Train boss baseline

* run `boss_baseline.ipynb`

### Voting

* python vote.py --csv pred_1.csv pred_2.csv ...


### hyperparamters

#### public score: 0.834
params =  {'hidden_layers': 7, 'hidden_dim': 256, 'reconstruct_alpha': 0, 'dropout': 0.4, 'weight_decay': 0.0001, 'lr': 0.002}

#### 0.83465
params =  {'hidden_layers': 5, 'hidden_dim': 256, 'reconstruct_alpha': 0, 'dropout': 0.4, 'weight_decay': 0.0001, 'lr': 0.002}

#### 0.82909
params =  {'hidden_layers': 5, 'hidden_dim': 256, 'reconstruct_alpha': 0, 'dropout': 0.4, 'weight_decay': 0.0001, 'lr': 0.002}

#### 0.83694
params =  {'hidden_layers': 5, 'hidden_dim': 256, 'reconstruct_alpha': 0.4, 'dropout': 0.4, 'weight_decay': 0.0001, 'lr': 0.002,'random_swap': 5}

#### voting

python .\vote.py --csv .\prediction_836.csv .\prediction_0.8345.csv .\prediction_0.834.csv .\prediction_0.829.csv 