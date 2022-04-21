import pandas as pd
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser()
parser.add_argument('--csv', nargs='+')
parser.add_argument('--out', default='vote.csv')
args = parser.parse_args()

values = []
for csv in args.csv:
    value = pd.read_csv(csv, index_col=0)
    values.append([k[0] for k in value.values.tolist()])
result = []
for i in range(len(values[0])):
    tmp = [values[j][i] for j in range(len(values))]
    result.append(Counter(tmp).most_common(1)[0][0])
with open(args.out, 'w') as f:
    f.write('ID,Answer\n')
    for i, y in zip(pd.read_csv(args.csv[0], index_col=0).index, result):
        if ',' in y:
            y = f'"{y}"'
        f.write('{},{}\n'.format(i, y))
for i in range(len(args.csv)):
    print(args.csv[i], sum([r != v for r, v in zip(result, values[i])]))