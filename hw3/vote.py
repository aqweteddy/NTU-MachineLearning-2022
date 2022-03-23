import pandas as pd
from argparse import ArgumentParser
import torch
from collections import Counter

parser = ArgumentParser()
parser.add_argument('--csv', nargs='+', type=str)
parser.add_argument('--out', default='voting.csv')
args = parser.parse_args()

dfs = []
for c in args.csv:
    dfs.append(pd.read_csv(c).values.tolist())
dfs = torch.tensor(dfs)[:,:, 1]
print(dfs.shape)
result = []
for i in range(dfs.shape[1]):
    result.append(Counter(dfs[:, i].tolist()).most_common(1)[0][0])
out = pd.read_csv(args.csv[0])
out['Category'] = result
cnt = {}
for c in args.csv:
    v = torch.tensor(pd.read_csv(c).values[:, 1].tolist())
    print(c, (torch.tensor(result) != v).sum().item())
out.to_csv(args.out, index=False)