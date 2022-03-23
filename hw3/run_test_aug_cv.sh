#!bin/bash
for d in $1/* ; do
    echo "$d"
    python test_aug.py  --ckpt $d/best.ckpt --model effnet_b5 --repeat 5 --weight 0.5 0.5 --out $d/submit.csv
done