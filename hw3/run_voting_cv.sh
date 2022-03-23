#!/bin/bash

csv=""
for d in $1/* ; do
    csv="${csv} ${d}/submit.csv"
done
echo $csv
python vote.py --csv $csv