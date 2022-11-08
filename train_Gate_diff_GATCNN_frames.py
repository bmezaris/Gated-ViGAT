import argparse
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datasets import FCVID, miniKINETICS, ACTNET
from model import ModelGCNConcAfterLocalFrame as Model_Basic_Local
from model import ModelGCNConcAfterGlobalFrame as Model_Basic_Global
from model import ModelGCNConcAfterClassifier as Model_Cls
from model import ExitingGatesGATCNN as Model_Gate

parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('vigat_model', nargs=1, help='Frame trained model')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='actnet', choices=['fcvid', 'minikinetics', 'actnet'])
parser.add_argument('--dataset_root', default='/m3/ActivityNet120', help='dataset root directory')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--milestones', nargs="+", type=int, default=[16, 35], help='milestones of learning decay')
parser.add_argument('--num_epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_objects', type=int, default=50, help='number of objects with best DoC')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--ext_method', default='VIT', choices=['VIT', 'RESNET'], help='Extraction method for features')
parser.add_argument('--resume', default=None, help='checkpoint to resume training')
parser.add_argument('--save_interval', type=int, default=10, help='interval for saving models (epochs)')
parser.add_argument('--save_folder', default='weights', help='directory to save checkpoints')
parser.add_argument('--cls_number', type=int, default=7, help='number of classifiers ')
parser.add_argument('--t_step', nargs="+", type=int, default=[1, 3, 6, 9, 15, 22, 30], help='Classifier frames')
parser.add_argument('--beta', type=float, default=1e-9, help='Multiplier of gating loss schedule')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()


def train_frame(model_cls, model_gate, model_vigat_local, model_vigat_global, dataset, loader, crit, crit_gate, opt, sched, device):
    epoch_loss = 0
    for i, batch in enumerate(loader):
        feats, feat_global, label, _ = batch

        feats = feats.to(device)
        feat_global = feat_global.to(device)
        label = label.to(device)

        feat_global_single, wids_frame_global = model_vigat_global(feat_global, get_adj=True)

        opt.zero_grad()
        feat_gate = feat_global_single
        feat_gate = feat_gate.unsqueeze(dim=1)
        loss_gate = 0.

        for t in range(args.cls_number):
            index_bestframes = torch.tensor(np.sort(np.argsort(wids_frame_global, axis=1)[:, -args.t_step[t]:])).to(device)
            feats_bestframes = feats.gather(dim=1, index=index_bestframes.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dataset.NUM_BOXES, dataset.NUM_FEATS)).to(device)
            feat_local_single = model_vigat_local(feats_bestframes)
            feat_single_cls = torch.cat([feat_local_single, feat_global_single], dim=-1)
            feat_gate = torch.cat((feat_gate, feat_local_single.unsqueeze(dim=1)), dim=1)
            out_data = model_cls(feat_single_cls)

            loss_t = crit(out_data, label).mean(dim=-1)
            e_t = args.beta * torch.exp(torch.tensor(t)/2.)
            labels_gate = loss_t < e_t
            out_data_gate = model_gate(feat_gate.to(device), t)
            loss_gate += crit_gate(out_data_gate, torch.Tensor.float(labels_gate).unsqueeze(dim=1))
        loss_gate = loss_gate/args.cls_number
        loss_gate.backward()
        opt.step()
        epoch_loss += loss_gate.item()

    sched.step()
    return epoch_loss / len(loader)


def main():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.dataset == 'fcvid':
        dataset = FCVID(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.BCEWithLogitsLoss(reduction='none')
        crit_gate = nn.BCEWithLogitsLoss()
    elif args.dataset == 'actnet':
        dataset = ACTNET(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.BCEWithLogitsLoss(reduction='none')
        crit_gate = nn.BCEWithLogitsLoss()
    elif args.dataset == 'minikinetics':
        dataset = miniKINETICS(args.dataset_root, is_train=True, ext_method=args.ext_method)
        crit = nn.CrossEntropyLoss(reduction='none')
        crit_gate = nn.BCEWithLogitsLoss()
    else:
        sys.exit("Unknown dataset!")

    device = torch.device('cuda:0')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    if args.verbose:
        print("running on {}".format(device))
        print("num samples={}".format(len(dataset)))
        print("missing videos={}".format(dataset.num_missing))

    start_epoch = 0
    # Gate Model
    model_gate = Model_Gate(args.gcn_layers, dataset.NUM_FEATS, num_gates=args.cls_number).to(device)
    opt = optim.Adam(model_gate.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)
    # Classifier Model
    model_cls = Model_Cls(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS).to(device)
    data_vigat = torch.load(args.vigat_model[0])
    model_cls.load_state_dict(data_vigat['model_state_dict'])
    model_cls.eval()
    # Vigat Model Local
    model_vigat_local = Model_Basic_Local(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS).to(device)
    model_vigat_local.load_state_dict(data_vigat['model_state_dict'])
    model_vigat_local.eval()
    # Vigat Model Global
    model_vigat_global = Model_Basic_Global(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS).to(device)
    model_vigat_global.load_state_dict(data_vigat['model_state_dict'])
    model_vigat_global.eval()

    model_gate.train()
    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.perf_counter()
        loss = train_frame(model_cls, model_gate, model_vigat_local, model_vigat_global, dataset, loader, crit, crit_gate, opt, sched, device)
        t1 = time.perf_counter()

        if (epoch + 1) % args.save_interval == 0:
            sfnametmpl = 'model_gate_samecls_gatcnn_diffplanes_diffet_1e-9-{}-{:03d}.pt'
            sfname = sfnametmpl.format(args.dataset, epoch + 1)
            spth = os.path.join(args.save_folder, sfname)
            torch.save({
                'epoch': epoch + 1,
                'loss': loss,
                'model_state_dict': model_gate.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'sched_state_dict': sched.state_dict()
            }, spth)
        if args.verbose:
            print("[epoch {}] loss={} dt={:.2f}sec".format(epoch + 1, loss, t1 - t0))


if __name__ == '__main__':
    main()
