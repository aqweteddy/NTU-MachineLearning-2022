import optuna
from train import run_train, set_seed
from argparse import Namespace


def objective(trial: optuna.trial.Trial) -> float:
    args = Namespace(
        batch_size=64,
        conv_kernel_size=trial.suggest_categorical('conv_kernel_size',
                                                   [16, 32]),
        d_model=trial.suggest_categorical('d_model', [40, 80, ]),
        data_dir='Dataset/',
        dropout=0.2,
        encoder_dim=trial.suggest_categorical('encoder_dim', [144, 256]),
        label_smoothing=trial.suggest_categorical('label_smoothing',
                                                  [0, 0.2, 0.4]),
        lr=trial.suggest_categorical('lr', [1e-3, 2e-3, 5e-4]),
        max_epochs=150,
        model_type='conformer',
        nhead=trial.suggest_categorical('nhead', [
            4,
            8,
        ]),
        num_layers=trial.suggest_categorical('num_layers', [1, 2, 3]),
        optimizer='adamw',
        pooling='attn',
        scheduler='warmup_cosine',
        warmup_epoch=trial.suggest_categorical('warmup_epoch', [1, 2, 3]))
    args.exp_name = f'conv_size_{args.conv_kernel_size}-d_model_{args.d_model}-encoder_dim_{args.encoder_dim}-lr_{args.lr}-nhead_{args.nhead}-n_layers_{args.num_layers}'
    # We optimize the number of layers, hidden units in each layer and dropouts.
    _, model = run_train(args)

    return model.best_val_acc

set_seed(1003)
pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
study = optuna.create_study(study_name='conformer',
                            direction="maximize",
                            pruner=pruner,
                            load_if_exists=True,
                            storage='sqlite:///tune.db')
study.optimize(objective, n_trials=50, timeout=50400)
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))