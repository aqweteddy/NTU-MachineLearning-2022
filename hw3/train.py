import torch
import pandas as pd
import wandb
import numpy as np
from ranger21 import Ranger21
from argparse import ArgumentParser
import os, random
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from tqdm import tqdm
from torch import nn
from dataset import FoodDataset, get_all_paths, get_transform, get_dataloader
from model import Model, FocalLoss, mixup
import numpy as np
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def run_train(args, train_loader, val_loader, device):
    model = Model(args.model).to(device)

    if args.loss == 'focal':
        criterion = FocalLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    else:
        raise NotImplementedError

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=1e-4,
                                    momentum=.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=1e-4,
                                     eps=1e-8)
    elif args.optimizer == 'ranger':
        optimizer = Ranger21(model.parameters(),
                             lr=args.lr,
                             weight_decay=1e-4,
                             num_batches_per_epoch=len(train_loader),
                             num_epochs=args.n_epochs,
                             num_warmup_iterations=20 * (len(train_loader)),
                             gc_conv_only=True)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=0.05,
                                      betas=(.9, .999))
    else:
        raise NotImplementedError

    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=12,
            threshold=0.0001,
            factor=0.1,
            verbose=True)
    elif args.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[80, 140],
                                                         gamma=0.1,
                                                         verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 20, 1e-7)
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise NotImplementedError

    stale = 0
    best_acc = 0
    wandb.init(project="ml-hw3",
               entity="tedli",
               config=args,
               name=args.exp_name)

    for epoch in range(args.n_epochs):
        model.train()

        # These are used to record information in training.
        train_loss = []
        for batch in tqdm(train_loader, leave=False):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            if epoch < args.n_epochs // 4 * 3 and args.mixup:
                imgs, labels = mixup(imgs, labels,
                                     random.randint(0, 4) / 10 + 1e-5)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 max_norm=10)
            wandb.log({'train/loss': loss.item()})
            # Update the parameters with computed gradients.
            optimizer.step()
            # Record the loss and accuracy.
            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(val_loader, leave=False):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(
                dim=-1) == labels.to(device).argmax(-1)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        if epoch > args.n_epochs // 4 * 3:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 20, 1e-6)

        if args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_acc)
        elif scheduler is None:
            pass
        else:
            scheduler.step()

        wandb.log({
            'val/acc': valid_acc,
            'val/loss': valid_loss,
            'lr': optimizer.state_dict()['param_groups'][0]['lr']
        })
        # update logs
        if valid_acc > best_acc:
            print(
                f"[ {epoch + 1:03d}/{args.n_epochs:03d} ] train_loss={train_loss:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f} -> best"
            )
        else:
            print(
                f"[ {epoch + 1:03d}/{args.n_epochs:03d} ] train_loss={train_loss:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}"
            )

        # save models
        if valid_acc > best_acc:
            torch.save(
                model.state_dict(), f"ckpt/{args.exp_name}/best.ckpt"
            )  # only save best to prevent output memory exceed error
            best_acc = valid_acc
            wandb.run.summary["best_accuracy"] = best_acc
            stale = 0
        else:
            stale += 1
            if stale > args.patients:
                print(
                    f"No improvment {args.patients} consecutive epochs, early stopping"
                )
                break


def run_test(args, test_tfm, device):

    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)

    test_set = FoodDataset(os.path.join(args.dataset_dir, "test"),
                           tfm=test_tfm)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True)
    model_best = Model(args.model).to(device)
    model_best.load_state_dict(torch.load(f"ckpt/{args.exp_name}/best.ckpt"))
    model_best.eval()
    prediction = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()
    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(1, len(test_set) + 1)]
    df["Category"] = prediction
    df.to_csv(f"ckpt/{args.exp_name}/submission.csv", index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--loss', default='focal')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--scheduler', default='none')
    parser.add_argument('--train_tfm', default='auto')

    parser.add_argument('--patients', default=100, type=int)
    parser.add_argument('--n_epochs', default=130, type=int)
    parser.add_argument('--gpuid', default='0')
    parser.add_argument('--cv', default=0, type=int)

    parser.add_argument('--dataset_dir', default='./food11')
    parser.add_argument('--exp_name', default='exp_name')
    args = parser.parse_args()
    myseed = 1002
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    random.seed(myseed)

    train_tfm, test_tfm = get_transform(args.train_tfm)

    all_img_paths = get_all_paths(os.path.join(
        args.dataset_dir, "training")) + get_all_paths(
            os.path.join(args.dataset_dir, "validation"))
    random.shuffle(all_img_paths)
    try:
        os.makedirs(f'ckpt/{args.exp_name}')
    except FileExistsError:
        pass

    if args.cv <= 0:
        train_path = all_img_paths[:int((len(all_img_paths) + 1) * .85)]
        val_path = all_img_paths[int((len(all_img_paths) + 1) * .85):]
        train_loader, val_loader = get_dataloader(train_path, val_path,
                                                  train_tfm, test_tfm,
                                                  args.batch_size)
        run_train(args, train_loader, val_loader, 'cuda')
        run_test(args, test_tfm, 'cuda')
    else:
        from sklearn.model_selection import KFold
        kf = KFold(args.cv, shuffle=True)
        base_exp_name = args.exp_name
        for k, (train_idx, val_idx) in enumerate(kf.split(all_img_paths)):
            train_path = [all_img_paths[i] for i in train_idx]
            val_path = [all_img_paths[i] for i in val_idx]
            train_loader, val_loader = get_dataloader(train_path, val_path,
                                                      train_tfm, test_tfm,
                                                      args.batch_size)
            args.exp_name = f'{base_exp_name}/cv{k}'
            try:
                os.makedirs(f'ckpt/{args.exp_name}')
            except FileExistsError:
                pass
            run_train(args, train_loader, val_loader, 'cuda')
