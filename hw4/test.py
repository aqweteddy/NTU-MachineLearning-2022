from argparse import ArgumentParser
from train import run_test, Model
from dataset import InferenceSegDataset
from pathlib import Path
import json, csv
from tqdm import tqdm
import torch
parser = ArgumentParser()
parser.add_argument('--ckpt')
parser.add_argument('--data_dir', default='Dataset/')
parser.add_argument('--out', default='submit.csv')
parser.add_argument('--mode', default='seg', help='seg or normal')
parser.add_argument('--seglen', default=128, type=int)
parser.add_argument('--batch', default=100, type=int)

args = parser.parse_args()

@torch.no_grad()
def run_test_seg(ckpt, seglen, datadir, out_path, device='cuda'):
    model = Model.load_from_checkpoint(ckpt).to(device)
    model.eval()
    dataset = InferenceSegDataset(datadir, seglen)
    mapping_path = Path(datadir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    speaker_num = len(mapping["id2speaker"])

    results = [["Id", "Category"]]
    # for feat_path, mel in tqdm(dataset)
    for i in tqdm(range(len(dataset))):
        mels = []
        for j in range(args.batch):
            feat_path, mel = dataset[i]
            mels.append(mel)
        mels = torch.stack(mels).to(device)
        outs = torch.softmax(model(mels), dim=-1)  #[B, num_classes]
        outs = outs.sum(0)
        pred = outs.argmax(-1).cpu().numpy()
        results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if args.mode == 'normal':
    print('mode', args.mode)
    run_test(args.ckpt, args.seglen, args.data_dir, args.out, 'cuda')
else:
    print('mode', args.mode)
    run_test_seg(args.ckpt, args.seglen, args.data_dir, args.out, 'cuda')