import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gumbel_sigmoid


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.norm = nn.LayerNorm(out_feats)
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, adj):
        x = x.matmul(self.weight)
        x = adj.matmul(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class GraphModule(nn.Module):
    def __init__(self, num_layers, num_feats):
        super().__init__()
        self.wq = nn.Linear(num_feats, num_feats)
        self.wk = nn.Linear(num_feats, num_feats)

        layers = []
        for i in range(num_layers):
            layers.append(GCNLayer(num_feats, num_feats))
        self.gcn = nn.ModuleList(layers)

    def forward(self, x, get_adj=False):
        qx = self.wq(x)
        kx = self.wk(x)
        dot_mat = qx.matmul(kx.transpose(-1, -2))
        adj = F.normalize(dot_mat.square(), p=1, dim=-1)

        for layer in self.gcn:
            x = layer(x, adj)

        x = x.mean(dim=-2)
        if get_adj is False:
            return x
        else:
            return x, adj


class ClassifierSimple(nn.Module):
    def __init__(self, num_feats, num_hid, num_class):
        super().__init__()
        self.fc1 = nn.Linear(num_feats, num_hid)
        self.fc2 = nn.Linear(num_hid, num_class)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class ModelGCNConcAfter(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(2 * num_feats, num_feats, num_class)

    def forward(self, feats, feat_global, get_adj=False):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)
        if get_adj is False:
            x = self.graph(feats)
            x = x.view(N, FR, -1)
            x = self.graph(x)
            y = self.graph(feat_global)
            x = torch.cat([x, y], dim=-1)
            x = self.cls(x)
            return x
        else:
            x, adjobj = self.graph(feats, get_adj)
            adjobj = adjobj.cpu()
            wids_objects = adjobj.numpy().sum(axis=1)
            x = x.view(N, FR, -1)

            x, adjframelocal = self.graph(x, get_adj)
            adjframelocal = adjframelocal.cpu()
            wids_frame_local = adjframelocal.numpy().sum(axis=1)

            y, adjframeglobal = self.graph(feat_global, get_adj)
            adjframeglobal = adjframeglobal.cpu()
            wids_frame_global = adjframeglobal.numpy().sum(axis=1)

            x = torch.cat([x, y], dim=-1)
            x = self.cls(x)

            return x, wids_objects, wids_frame_local, wids_frame_global


class ModelGCNConcAfterGlobalOnly(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feat_global): #(,feats,)

        x = self.graph(feat_global)
        x = self.cls(x)

        return x


class ModelGCNConcAfterLocalOnly(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feats): #,feat_global
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)

        x = self.graph(feats)
        x = x.view(N, FR, -1)
        x = self.graph(x)
        x = self.cls(x)

        return x


class ModelGATPolicyDeterministic(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(2 * num_feats, num_feats, num_class)

    def forward(self, feat_global, get_adj=True):
        y, adjframeglobal = self.graph(feat_global, get_adj)
        adjframeglobal = adjframeglobal.cpu()
        wids_frame_global = adjframeglobal.detach().numpy().sum(axis=1)
        # y = self.cls(y, device)

        return wids_frame_global


class ModelGATPolicy(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(2 * num_feats, num_feats, num_class)

    def forward(self, feat_global, get_adj=True):
        y, adjframeglobal = self.graph(feat_global, get_adj)
        wids_frame_global = adjframeglobal.sum(axis=1)

        return wids_frame_global


class ModelTotalGATPolicy(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.vigat = ModelGCNConcAfter(gcn_layers, num_feats, num_class)
        self.ModelGATPolicy = ModelGATPolicy(gcn_layers, num_feats, num_class)

    def forward(self, feats, feat_global, temp, gs_thresh):
        wids_frame_global = self.ModelGATPolicy(feat_global)
        # Policy Function
        # TODO CHECK NON-ZERO Wids
        mask = gumbel_sigmoid(logits=torch.log(wids_frame_global), temperature=temp, thresh=gs_thresh, hard=True)
        # Index Selection
        kept_feats = feats * mask.unsqueeze(-1).unsqueeze(-1)

        out_data = self.vigat(kept_feats, feat_global)
        return out_data, mask


class ModelTotalGATPolicyHead(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class, num_frames):
        super().__init__()
        self.vigat = ModelGCNConcAfter(gcn_layers, num_feats, num_class)
        self.ModelGATPolicy = ModelGATPolicy(gcn_layers, num_feats, num_class)
        self.fc = nn.ReLU(nn.Linear(num_frames, num_frames))
        self.drop = nn.Dropout()

    def forward(self, feats, feat_global, temp, gs_thresh):
        wids_frame_global = self.ModelGATPolicy(feat_global)
        wids_frame_global = self.fc(wids_frame_global)
        wids_frame_global = self.drop(wids_frame_global)
        # Policy Function
        # TODO CHECK NON-ZERO Wids
        mask = gumbel_sigmoid(logits=wids_frame_global, temperature=temp, thresh=gs_thresh, hard=True)
        # Index Selection
        kept_feats = feats * mask.unsqueeze(-1).unsqueeze(-1)

        out_data = self.vigat(kept_feats, feat_global)
        return out_data, mask


class ModelGCNConcAfterGlobal(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feat_global):

        y = self.graph(feat_global)
        x = self.cls(y)

        return x, y


class ModelGCNConcAfterFrame(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_frame_classifiers, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.classifiers = nn.ModuleList()
        self.num_frame_classifiers = num_frame_classifiers
        for m in range(0, self.num_frame_classifiers):
            self.classifiers.append(ClassifierSimple(2 * num_feats, num_feats, num_class))

    def forward(self, feats, feat_global_single, feat_single_previous, t_c=torch.tensor(0), get_adj=False):
        N, FR, B, NF = feats.shape
        feats = feats.reshape(N * FR, B, NF)  # .view  .reshape .contiguous()
        if get_adj is False:
            x = self.graph(feats)
            x = x.view(N, FR, -1)
            x = self.graph(x)
            y = torch.cat([x, feat_global_single], dim=-1)
            x = self.classifiers[t_c](y)
            return x, y
        else:
            x, adjobj = self.graph(feats, get_adj)
            adjobj = adjobj.cpu()
            wids_objects = adjobj.numpy().sum(axis=1)
            x = x.view(N, FR, -1)

            x, adjframelocal = self.graph(x, get_adj)
            adjframelocal = adjframelocal.cpu()
            wids_frame_local = adjframelocal.numpy().sum(axis=1)

            y = torch.cat([x, feat_global_single], dim=-1)
            x = self.classifiers[t_c](y)

            return x, y, wids_objects, wids_frame_local


class ExitingGate(nn.Module):
    def __init__(self, in_planes):
        super(ExitingGate, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 128, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(128, 1, bias=True)

    def forward(self, x0, x1, force_hard=True, prev_features=None):
        x0 = F.relu(self.bn1(self.conv1(x0)))
        x0 = F.relu(self.bn2(self.conv2(x0)))
        x0 = torch.flatten(x0, 1)
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = torch.flatten(x1, 1)
        x = torch.cat([x0, x1], dim=1)
        out = self.linear(x)
        out = self.sigmoid(out)
        # out[out >= 0.5] = 1
        # out[out < 0.5] = 0

        return out


class ExitingGates(nn.Module):
    def __init__(self, in_planes, num_gates):
        super(ExitingGates, self).__init__()
        self.exiting_gates = nn.ModuleList()
        self.num_gates = num_gates
        for m in range(0, self.num_gates):
            self.exiting_gates.append(ExitingGate(in_planes))

    def forward(self, x0, x1, gate_num, force_hard=True, prev_features=None):
        out = self.exiting_gates[gate_num](x0, x1)
        return out


class ExitingGateGAT(nn.Module):
    def __init__(self,  gcn_layers, num_feats):
        super(ExitingGateGAT, self).__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):
        y = self.graph(feat)
        x = self.cls(y)
        out = self.sigmoid(x)

        return out


class ExitingGatesGAT(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_gates):
        super(ExitingGatesGAT, self).__init__()
        self.exiting_gates = nn.ModuleList()
        self.num_gates = num_gates
        for m in range(0, self.num_gates):
            self.exiting_gates.append(ExitingGateGAT(gcn_layers, num_feats))

    def forward(self, feat, gate_num):
        out = self.exiting_gates[gate_num](feat)
        return out


class ModelClassifier(nn.Module):
    def __init__(self, num_feats, num_class, num_frame_classifiers):
        super(ModelClassifier, self).__init__()
        self.classifiers = nn.ModuleList()
        self.num_frame_classifiers = num_frame_classifiers
        for m in range(0, self.num_frame_classifiers):
            self.classifiers.append(ClassifierSimple(2 * num_feats, num_feats, num_class))

    def forward(self, y, t_c=torch.tensor(0)):
        x = self.classifiers[t_c](y)
        return x


class ExitingGateGATCNN(nn.Module):
    def __init__(self, gcn_layers, in_planes, out_planes=256, inter_planes=512):
        super(ExitingGateGATCNN, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.inter_planes = inter_planes
        self.graph = GraphModule(gcn_layers, self.out_planes)
        self.cls = ClassifierSimple(self.out_planes, int(self.out_planes / 2), 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.in_planes, self.inter_planes, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(self.inter_planes, self.out_planes, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inter_planes)
        self.bn2 = nn.BatchNorm2d(self.out_planes)

    def forward(self, feat):
        N, FR, NF = feat.shape
        feat = feat.view(N * FR, NF, 1, 1)
        feat = F.relu(self.bn1(self.conv1(feat)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        feat = feat.view(N, FR, self.out_planes)
        y = self.graph(feat)
        x = self.cls(y)
        out = self.sigmoid(x)

        return out


class ExitingGatesGATCNN(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_gates):
        super(ExitingGatesGATCNN, self).__init__()
        self.exiting_gates = nn.ModuleList()
        self.num_gates = num_gates
        for m in range(0, self.num_gates):
            self.exiting_gates.append(ExitingGateGATCNN(gcn_layers, in_planes=num_feats))

    def forward(self, feat, gate_num):
        out = self.exiting_gates[gate_num](feat)
        return out


class ModelGCNConcAfterGlobalFrame(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(2 * num_feats, num_feats, num_class)

    def forward(self, feat_global, get_adj=False):
        if get_adj is False:
            y = self.graph(feat_global)
            return y
        else:
            y, adjframeglobal = self.graph(feat_global, get_adj)
            adjframeglobal = adjframeglobal.cpu()
            wids_frame_global = adjframeglobal.detach().numpy().sum(axis=1)

            return y, wids_frame_global


class ModelGCNConcAfterLocalFrame(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(2 * num_feats, num_feats, num_class)

    def forward(self, feats, get_adj=False):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)
        if get_adj:
            x, adjobj = self.graph(feats, get_adj)
            adjobj = adjobj.cpu()
            wids_objects = adjobj.numpy().sum(axis=1)
            x = x.view(N, FR, -1)

            x, adjframelocal = self.graph(x, get_adj)
            adjframelocal = adjframelocal.cpu()
            wids_frame_local = adjframelocal.numpy().sum(axis=1)

            return x, wids_objects, wids_frame_local
        else:
            x = self.graph(feats)
            x = x.view(N, FR, -1)
            x = self.graph(x)

        return x


class ModelGCNConcAfterClassifier(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(2 * num_feats, num_feats, num_class)

    def forward(self, feats):
        x = self.cls(feats)
        return x
