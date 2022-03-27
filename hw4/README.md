# HW4

* organize TA's example code.
* conformer code: copy from `torchaudio` package
* use `wandb` to monitor training process
* use `pytorch_lightning` to rewrite training.

## boss baseline training

* public score: 0.8745

### training

```sh
python train.py --model_type conformer --optimizer adamw --pooling attn --amsoftmax --cv 5
```

### testing

```sh
python test.py --ckpt [path_to_cv1_best_checkpoints] --out cv1.csv
python test.py --ckpt [path_to_cv2_best_checkpoints] --out cv2.csv
python test.py --ckpt [path_to_cv3_best_checkpoints] --out cv3.csv
python test.py --ckpt [path_to_cv4_best_checkpoints] --out cv4.csv
python test.py --ckpt [path_to_cv5_best_checkpoints] --out cv5.csv
python vote.py python vote.py --csv cv3.csv cv5.csv cv2.csv cv4.csv cv1.csv  # order is assigned by the validation score in decreasing.
```

### other arguments

```sh
python train.py --help
```