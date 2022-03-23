# README

## code

* Adapt code from NTU ML example.
* Use `wandb` to visualize the experiemtal result.

## passed baseline

* boss baseline:  0.91035

## reproduce

### train

* efficientnet b5
*  5 fold CV
```sh
train.py --model effnet_b5 --loss focal --mixup --lr 5e-3 --scheduler ReduceLROnPlateau --exp_name effnet-b5-cv5 --cv 5 --gpuid 0
```

### test augmentation

```sh
sh run_test_aug_cv.sh ckpt/effnet-b5-cv5/
```

### voting

```sh
sh run_voting_cv.sh ckpt/effnet-b5-cv5
```
* result: `voting.csv`