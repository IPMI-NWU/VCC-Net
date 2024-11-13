import datetime
import argparse
import os
import random
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
from a_Data import misc
from b_Classification.dataset import build_dataset
from b_Classification.models.models_list import vit_s, resnet18, resnet50, resnet101, swin_s
from b_Classification.models.vig import pvig_s_224_gelu


def get_args_parser():
    parser = argparse.ArgumentParser('VCGGNN classification', add_help=False)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--output_dir', default='output/TB_exp_Res50_e4/', help='path where to save')
    parser.add_argument('--dataset_name', default='TB', help='SIIM, MIMIC, TB, COVID')
    parser.add_argument('--label_num', default=2, help='2, 3, 2, 3', type=int)
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    return parser

def main(args):
    writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(args.device)

    # model = resnet18(args.label_num)
    model = resnet50(args.label_num)
    # model = resnet101(args.label_num)
    # model = vit_s(args.label_num)
    # model = swin_s(args.label_num)
    # model = pvig_s_224_gelu(args.label_num)
    # save_model = torch.load('models/pretrained_models/pvig_s_82.1.pth.tar')
    # model_dict = model.state_dict()
    # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)

    output_dir = Path(args.output_dir)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)
    print('Building training dataset...')
    dataset_train = build_dataset(args.dataset_name, 'train', args.size)
    print('Number of training images: {}'.format(len(dataset_train)))
    print('Building validation dataset...')
    dataset_val = build_dataset(args.dataset_name, 'test', args.size)
    print('Number of validation images: {}'.format(len(dataset_val)))
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers)
    # -------------------------------------------------------
    print("Start training")
    start_time = time.time()
    best_acc = None
    print_freq = 50
    for epoch in range(0, args.epochs):
        print('-' * 40)
        print('Epoch: [{}] '.format(epoch + 1))
        print('Training...')
        # ------------------------------------------------------------
        # train
        # ------------------------------------------------------------
        model.train()
        train_start_time = time.time()
        step = 0
        pred_list = []
        label_list = []
        criterion_cls = nn.CrossEntropyLoss()
        for img, gaze, label, _ in dataloader_train:
            # ------------------------------------------------------------
            img = img.to(device)
            label = label.to(device)
            # ------------------------------------------------------------
            output = model(img)
            # ------------------------------------------------------------
            loss_cls = criterion_cls(output, label)
            loss = loss_cls
            # ------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ------------------------------------------------------------
            _, pred = torch.max(output, 1)
            pred_list.append(pred.cpu().detach().numpy().tolist())
            label_list.append(label.cpu().detach().numpy().tolist())
            if step % print_freq == 0:
                print('    lr: {:.6f}'.format(optimizer.param_groups[0]["lr"]))
                print('    loss_cls: {:.4f}'. format(loss_cls.item()))
            step = step + 1
        # ------------------------------------------------------------
        train_total_time = time.time() - train_start_time
        total_time_str = str(datetime.timedelta(seconds=int(train_total_time)))
        pred_list = [b for a in pred_list for b in a]
        label_list = [b for a in label_list for b in a]
        cm = confusion_matrix(pred_list, label_list)
        print(cm)
        print('Training time: {} ({:.4f} r / it )'.format(total_time_str, train_total_time / len(dataloader_train)))
        writer.add_scalar('loss', loss.item(), epoch)
        lr_scheduler.step()
        # ------------------------------------------------------------
        # evaluate
        # ------------------------------------------------------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            pred_list = []
            label_list = []
            print('Val...')
            val_start_time = time.time()
            for img, gaze, label, _ in dataloader_val:
                # ------------------------------------------------------------
                img = img.to(device)
                label = label.to(device)
                # ------------------------------------------------------------
                output = model(img)
                loss = criterion_cls(output, label)
                # ------------------------------------------------------------
                _, pred = torch.max(output, 1)
                pred_list.append(pred.cpu().detach().numpy().tolist())
                label_list.append(label.cpu().detach().numpy().tolist())
            # ------------------------------------------------------------
            val_total_time = time.time() - val_start_time
            total_time_str = str(datetime.timedelta(seconds=int(val_total_time)))
            print('Val time: {} ({:.4f} r / it )'.format(total_time_str, val_total_time / len(dataloader_val)))
            pred_list = [b for a in pred_list for b in a]
            label_list = [b for a in label_list for b in a]
            acc_score = accuracy_score(pred_list, label_list)
            print('Val Acc: {:.4f}  Val loss: {:.4f}'.format(acc_score, loss.item()))
            cm = confusion_matrix(pred_list, label_list)
            print(cm)
            writer.add_scalar('val_loss', loss.item(), epoch)
            writer.add_scalar('val_acc', acc_score, epoch)
            # ------------------------------------------------------------
            # save checkpoint for high dice score
            # ------------------------------------------------------------
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if best_acc is None or acc_score > best_acc:
                    best_acc = acc_score
                    print("Update best model!")
                    checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                # You can change the threshold
                if acc_score > 0.60:
                    print("Update high dice score model!")
                    file_name = str(acc_score)[0:6] + '_' + str(epoch + 1) + '_checkpoint.pth'
                    checkpoint_paths.append(output_dir / file_name)
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    misc.save_on_master({
                        'model': model.state_dict(),
                    }, checkpoint_path)
        print()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classification training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)