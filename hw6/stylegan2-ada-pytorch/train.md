# HW6

* I use `NVlabs/StyleGAN2-ADA; Official PyTorch implementation` to train this homework. ([repo](https://github.com/NVlabs/stylegan2-ada-pytorch))

## preprocess data
```bash
python dataset_tool.py --source ../faces --dest=128.zip --width=128 --height=128
```

## train
```bash
python train.py --outdir=training-runs --data 128.zip --gpus=1 --batch 64 --workers 10 --kimg 1000
```
## test

* choose 600 kimgs checkpoint.

```bash
python generate.py --network training-runs/[path to 600 kimgs checkpoint] --seeds 1-1000 --outdir output --disc_threshold 1.
cd output
tar -zcf ../submission.tgz *.jpg
cd ..
```