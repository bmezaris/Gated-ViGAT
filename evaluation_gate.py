import argparse
import time
import torch
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, top_k_accuracy_score
import numpy as np
from datasets import miniKINETICS, ACTNET
from model import ModelGCNConcAfterLocalFrame as Model_Basic_Local
from model import ModelGCNConcAfterGlobalFrame as Model_Basic_Global
from model import ModelGCNConcAfterClassifier as Model_Cls
from model import ExitingGatesGATCNN as Model_Gate

parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('vigat_model', nargs=1, help='Vigat trained model')
parser.add_argument('gate_model', nargs=1, help='Gate trained model')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='actnet', choices=['minikinetics', 'actnet'])
parser.add_argument('--dataset_root', default='/ActivityNet', help='dataset root directory')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_objects', type=int, default=50, help='number of objects with best DoC')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--ext_method', default='VIT', choices=['VIT'], help='Extraction method for features')
parser.add_argument('--save_scores', action='store_true', help='save the output scores')
parser.add_argument('--save_path', default='scores.txt', help='output path')
parser.add_argument('--cls_number', type=int, default=7, help='number of classifiers ')
parser.add_argument('--t_step', nargs="+", type=int, default=[1, 3, 6, 9, 15, 22, 30], help='Classifier frames')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()


def evaluate(model_gate, model_cls, model_vigat_local, model_vigat_global, dataset, loader, scores, class_of_video, class_vids, device):
    gidx = 0
    class_selected = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats, feat_global, _, _ = batch

            feats = feats.to(device)
            feat_global = feat_global.to(device)
            feat_global_single, wids_frame_global = model_vigat_global(feat_global, get_adj=True)
            feat_gate = feat_global_single
            feat_gate = feat_gate.unsqueeze(dim=1)

            for t in range(args.cls_number):
                index_bestframes = torch.tensor(np.sort(np.argsort(wids_frame_global, axis=1)[:, -args.t_step[t]- 1:])).to(device)
                feats_bestframes = feats.gather(dim=1, index=index_bestframes.unsqueeze(-1).unsqueeze(-1).
                                                expand(-1, -1, dataset.NUM_BOXES, dataset.NUM_FEATS)).to(device)
                feat_local_single = model_vigat_local(feats_bestframes)
                feat_single_cls = torch.cat([feat_local_single, feat_global_single], dim=-1)
                feat_gate = torch.cat((feat_gate, feat_local_single.unsqueeze(dim=1)), dim=1)

                out_data = model_cls(feat_single_cls)

                out_data_gate = model_gate(feat_gate.to(device), t)
                class_selected = t
                exit_switch = out_data_gate >= 0.5
                if exit_switch or t == (args.cls_number-1):
                    class_vids[t] += 1
                    break

            shape = out_data.shape[0]
            class_of_video[gidx:gidx + shape] = class_selected
            scores[gidx:gidx + shape, :] = out_data.cpu()
            gidx += shape
        return class_vids


def main():
    if args.dataset == 'actnet':
        dataset = ACTNET(args.dataset_root, is_train=False, ext_method=args.ext_method)
    elif args.dataset == 'minikinetics':
        dataset = miniKINETICS(args.dataset_root, is_train=False, ext_method=args.ext_method)
    else:
        sys.exit("Unknown dataset!")
    device = torch.device('cuda:0')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.verbose:
        print("running on {}".format(device))
        print("num samples={}".format(len(dataset)))
        print("missing videos={}".format(dataset.num_missing))

    # Gate Model
    model_gate = Model_Gate(args.gcn_layers, dataset.NUM_FEATS, num_gates=args.cls_number).to(device)
    data_gate = torch.load(args.gate_model[0])
    model_gate.load_state_dict(data_gate['model_state_dict'])
    model_gate.eval()
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

    num_test = len(dataset)
    scores = torch.zeros((num_test, dataset.NUM_CLASS), dtype=torch.float32)
    class_of_video = torch.zeros((num_test),dtype=torch.int)
    class_vids = torch.zeros(args.cls_number)

    t0 = time.perf_counter()
    evaluate(model_gate, model_cls, model_vigat_local, model_vigat_global, dataset, loader, scores, class_of_video, class_vids, device)
    t1 = time.perf_counter()

    # Change tensors to 1d-arrays
    scores = scores.numpy()
    class_of_video = class_of_video.numpy()
    class_vids = class_vids.numpy()
    num_total_vids = int(np.sum(class_vids))
    assert num_total_vids == len(dataset)
    class_vids_rate = class_vids/num_total_vids
    avg_frames = int(np.sum(class_vids_rate*args.t_step))

    if args.dataset == 'actnet':
        ap = average_precision_score(dataset.labels, scores)
        class_ap = np.zeros(args.cls_number)
        for t in range(args.cls_number):
            if sum(class_of_video == t) == 0:
                print('No Videos fetched by classifier {}'.format(t))
                continue
            current_labels = dataset.labels[class_of_video == t, :]
            current_scores = scores[class_of_video == t, :]
            columns_to_delete = []
            for l in range(current_labels.shape[1]):
                if sum(current_labels[:, l]) == 0:
                    columns_to_delete.append(l)
            current_labels = np.delete(current_labels, columns_to_delete, 1)
            current_scores = np.delete(current_scores, columns_to_delete, 1)
            class_ap[t] = average_precision_score(current_labels, current_scores)

            # class_ap[t] = average_precision_score(dataset.labels[class_of_video == t, :], scores[class_of_video == t, :], average='samples')
        for t in range(args.cls_number):
            print('Classifier {}: top1={:.2f}% Cls frames:{}'.format(t, 100 * class_ap[t], args.t_step[t]))
        print('top1={:.2f}% dt={:.2f}sec'.format(100 * ap, t1 - t0))
        print('Total Exits per Classifier: {}'.format(class_vids))
        print('Average Frames taken: {}'.format(avg_frames))
    elif args.dataset == 'minikinetics':
        top1 = top_k_accuracy_score(dataset.labels, scores, k=1)
        top5 = top_k_accuracy_score(dataset.labels, scores, k=5)
        print('top1 = {:.2f}%, top5 = {:.2f}% dt = {:.2f}sec'.format(100 * top1, 100 * top5, t1 - t0))


if __name__ == '__main__':
    main()
