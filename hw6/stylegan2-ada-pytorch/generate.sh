#!bin/bash
echo loaded weight: $1, save_dir: $2, discriminator loss threshold: $3

rm -rf $2
python generate.py --network $1 --seeds 1-1000 --outdir $2 --disc_threshold $3
rm -f submission.tgz
cd $2
pwd
wait
tar -zcf ../submission.tgz *.jpg