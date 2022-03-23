import os
from argparse import ArgumentParser
from dataset import get_transform, FoodDataset, get_all_paths
from model import Model
import torch
from tqdm import tqdm
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--model', default='effnet_b5')
parser.add_argument('--repeat', type=int)
parser.add_argument('--weight', nargs='+', type=float)
parser.add_argument('--out', default='test_aug_submit.csv')
parser.add_argument('--dataset_dir', default='./food11')
parser.add_argument('--gpuid', default='0')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

train_tfm, test_tfm = get_transform('auto')
test_img_paths = get_all_paths(os.path.join(args.dataset_dir, "test"))
test_ds = FoodDataset(test_img_paths, tfm=None)


def batch_img(img, test_tfm, train_tfm, repeat_times=5):
    """
    oriimg, rand_transform 1, .. 2
    """
    result = [test_tfm(img)]
    for i in range(repeat_times):
        result.append(train_tfm(img))
    return torch.stack(result)

model_best = Model(args.model).to('cuda')
model_best.load_state_dict(torch.load(args.ckpt))
model_best.eval()
prediction = []

tfm_weight = args.weight[1] / args.repeat
WEIGHTS = torch.tensor([args.weight[0]] + [tfm_weight] * args.repeat).to('cuda')
print(WEIGHTS)
with torch.no_grad():
    for data,_ in tqdm(test_ds):
        batch = batch_img(data, test_tfm, train_tfm, args.repeat)
        test_pred = model_best(batch.to('cuda'))
        pred = torch.softmax(test_pred, dim=-1)
        pred = WEIGHTS.matmul(pred).reshape(-1)
        # [transform_num, num_classes]
        pred = pred.argmax().item()
        prediction.append(pred)


def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_ds)+1)]
df["Category"] = prediction
df.to_csv(args.out,index = False)