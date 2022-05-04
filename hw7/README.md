# HW7

## train boss baseline

* combine training data and validation data to do 5-fold CV 
* data augmentation:
    * apply random mask(mask random tokens to \[MASK\])
    * apply random swap words
* apply fp16 training

```sh
pip install -r requirements.txt # install required package
python train_kfold.py --kfold 5 --gpuid 0 --gpus 1 --exp_name macbert_large-ft_mrc_maskprob.15 --weight_decay 1e-2 --batch_size 8 --accumulate_grad_batches 8 --pretrained luhua/chinese_pretrain_mrc_macbert_large --lr 3e-5 --scheduler cosine_warmup --warmup_epochs 1 --min_epochs 1 --max_epoch 7 --precision 16 --mask_prob 0.15
```

## inference

```sh
run_test_vote.py --qa_ckpt CKPT0_PATH CKPT1_PATH CKPT2_PATH CKPT3_PATH CKPT4_PATH --output vote.csv --gpuid 0
```
