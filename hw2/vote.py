from ast import parse
import pandas as pd
from argparse import ArgumentParser
import torch
from collections import Counter
# python .\vote.py --csv .\prediction_0.834.csv .\prediction_0.829.csv .\prediction0.826.csv
parser = ArgumentParser()
parser.add_argument('--csv', nargs='+', type=str, help='important -> not important')
parser.add_argument('--out', default='vote.csv')
args = parser.parse_args()

values = []
for csv in args.csv:
    value = pd.read_csv(csv, index_col=0)
    values.append(value.values.tolist())

values = torch.tensor(values).reshape(len(args.csv), -1) # [NUM_FILES, NUM_CASE]
result = []
for i in range(values.shape[-1]):
    result.append(Counter(values[:, i].tolist()).most_common(1)[0][0])

with open(args.out, 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(result):
        f.write('{},{}\n'.format(i, y))