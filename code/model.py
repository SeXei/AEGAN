import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import math

class NodeInteraction(nn.Module):
    def __init__(self, edge_dim, dropout):
        super(NodeInteraction,self).__init__()
        self.W = nn.Linear(2*edge_dim, 1)
        self.edge_dim = edge_dim
        self.W_in = nn.Linear(edge_dim, edge_dim)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, edge):
        batch, edges, edge_dim = edge.shape
        n = int(np.sqrt(edges))
        edge = self.dropout(F.relu(self.W_in(edge)))
        merge = torch.cat([edge, edge.reshape(-1, n, n, self.edge_dim).permute(0, 2, 1, 3).reshape(-1, edges, edge_dim)],dim=-1)
        o = self.W(merge)
        return torch.sigmoid(o)
    
class NodeGAT(nn.Module):
    def __init__(self, Node_in_features, out_features, dropout, alpha):
        super(NodeGAT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.Node_in_features = Node_in_features
        self.out_features = out_features
        self.W = nn.Linear(self.Node_in_features, self.out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.A = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.A, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
    def forward(self, Nodes, o):
        n = Nodes.shape[1]
        h = self.dropout(self.W(Nodes))
        assert not torch.isnan(h).any()
        e = self._prepare_attentional_mechanism_input(h)
        assert not torch.isnan(e).any()
#         g = gate.reshape(-1, n, self.out_features)
        attention = F.softmax(e*o, dim=-1)
        h_prime = torch.matmul(attention, h)
        assert not torch.isnan(h_prime).any()
        return self.dropout(self.leakyrelu(h_prime))
    def _prepare_attentional_mechanism_input(self, Wh):
        batchsize = Wh.shape[0]
        nodes = Wh.shape[1]
        Wh1 = torch.matmul(Wh, self.A[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.A[self.out_features: , :])
        # # broadcast add
        e = Wh1 + Wh2.transpose(-1, -2)
        return self.dropout(self.leakyrelu(e))
    
class NodeMultiheadGAT(nn.Module):
    def __init__(self, Node_in_features, out_features, dropout, alpha, head):
        super(NodeMultiheadGAT, self).__init__()
        self.heads = nn.ModuleList([NodeGAT(Node_in_features, out_features, dropout, alpha) for _ in range(head)])
        self.out = nn.Linear(out_features*head, out_features)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, Nodes, adj, gate):
        b, n, d = Nodes.shape
        g = gate.reshape(b, n, n)
        o = (adj+g)*g
        output = torch.cat([m(Nodes, o) for m in self.heads], dim=-1)
        return self.dropout(self.leakyrelu(self.out(output))), o

class EdgeTransformer(nn.Module):
    def __init__(self, edgefea:int, dropout=0.5):
        super(EdgeTransformer, self).__init__()
        self.edgefea = edgefea
        self.Wg2 = nn.Linear(self.edgefea, self.edgefea) # gate
        self.Wq = nn.Linear(self.edgefea, self.edgefea) # query
        self.Wk = nn.Linear(self.edgefea, self.edgefea) # key
        self.Wv = nn.Linear(self.edgefea, self.edgefea) # value
        self.Wb = nn.Linear(self.edgefea, 1) # bi-
        self.Wo = nn.Linear(self.edgefea, self.edgefea) # out
#         self.layernorm = nn.LayerNorm(self.edgefea)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, Edges):
#         Edges = self.layernorm(Edges)
        fea_dk = Edges.shape[-1] 
        edgesnum = int(math.sqrt(Edges.shape[1]))
#         ab = (self.Wt(Edges) * torch.sigmoid(self.Wg1(Edges))).reshape(-1, edgesnum, edgesnum, fea_dk)
        q = self.dropout(self.Wq(Edges).reshape(-1, edgesnum, edgesnum, fea_dk))
        k = self.dropout(self.Wk(Edges).reshape(-1, edgesnum, edgesnum, fea_dk))
        v = self.dropout(self.Wv(Edges).reshape(-1, edgesnum, edgesnum, fea_dk))
        b = self.dropout(self.Wb(Edges))
        g = self.dropout(torch.sigmoid(self.Wg2(Edges)))
        alpha = (torch.matmul(q, k.transpose(-2, -1)) + (torch.matmul(q.reshape(-1, edgesnum, edgesnum*fea_dk)\
                .transpose(-2, -1).reshape(-1, edgesnum, fea_dk, edgesnum).transpose(-2, -1), k.reshape(-1, edgesnum,\
                edgesnum*fea_dk).transpose(-2, -1).reshape(-1, edgesnum, fea_dk, edgesnum))).reshape(-1, edgesnum, \
                edgesnum*edgesnum).transpose(-2, -1).reshape(-1, edgesnum, edgesnum, edgesnum).transpose(-2, -1))/\
                math.sqrt(fea_dk) + (b.reshape(-1, edgesnum, 1, edgesnum) + b.reshape(-1, edgesnum, edgesnum)\
                .transpose(-2, -1).reshape(-1, edgesnum, 1, edgesnum)) # q[i,j](k[i,t]+k[t,j])/fea_dk^(1/2)+b[i,t]+b[t,j]
        alpha = self.dropout(F.softmax(alpha, dim=-1)) 
        # 汇聚
        out = g * (torch.matmul(alpha, v) + torch.matmul(alpha.reshape(-1, edgesnum, edgesnum*edgesnum).transpose(-2, \
              -1).reshape(-1, edgesnum, edgesnum, edgesnum).transpose(-2, -1), v.reshape(-1, edgesnum, edgesnum*fea_dk)\
              .transpose(-2, -1).reshape(-1, edgesnum, fea_dk, edgesnum).transpose(-2, -1))).\
              reshape(-1, edgesnum*edgesnum, fea_dk)
        out = self.dropout(F.relu(self.Wo(out)))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        return out
    
class EdgeUpdate(nn.Module):
    def __init__(self, nodefeat, edgefeat, dropout):
        super(EdgeUpdate, self).__init__()
        self.edgefeat = edgefeat
        self.dropout = nn.Dropout(p=dropout)
        self.Q = nn.Linear(nodefeat, edgefeat, bias=False)
        nn.init.xavier_uniform_(self.Q.weight, gain=1.414)
        self.A = nn.Parameter(torch.empty(size=(2 * edgefeat, 1)))
        nn.init.xavier_uniform_(self.A, gain=1.414)
        # self.out = nn.Linear()
    def forward(self, nodes, edges):
        q = self.Q(nodes)  # [batchsize, nodes, edg_feat]
        assert not torch.isnan(q).any()
        k = self._nodes_edges_att_(q, edges)
        assert not torch.isnan(k).any()
        return k + edges
    def _nodes_edges_att_(self, q, k):
        batchsize = k.shape[0]
        nodes = q.shape[1]
        w1 = torch.matmul(q, self.A[:self.edgefeat, :])  # [batchsize, nodes, 1]
        w2 = torch.matmul(k, self.A[self.edgefeat:, :]).reshape(batchsize, nodes, nodes)  # [batchsize, nodes, nodes]
        w = torch.softmax(w1 + w2, dim=-1).reshape(batchsize, nodes, nodes, 1)  # [batchsize, nodes, nodes, 1]
        v = q.reshape(batchsize, nodes, 1, self.edgefeat)  # [batchsize, nodes, 1, edgefeat]
        k = torch.mul(w, v).reshape(batchsize, nodes * nodes, self.edgefeat)  # [batchsize, nodes, nodes, edgefeat] -> [batchsize, nodes*nodes, edgefeat]
        return self.dropout(F.relu(k))

class NodeUpdate(nn.Module):
    def __init__(self, nodefeat, edgefeat, dropout):
        super(NodeUpdate, self).__init__()
        self.nodefeat = nodefeat
        self.dropout = nn.Dropout(p=dropout)
        self.Q = nn.Linear(edgefeat, nodefeat, bias=False)
        nn.init.xavier_uniform_(self.Q.weight, gain=1.414)
        self.A = nn.Parameter(torch.empty(size=(2 * nodefeat, 1)))
        nn.init.xavier_uniform_(self.A, gain=1.414)
    def forward(self, nodes, edges):
        q = self.dropout(self.Q(edges))
        assert not torch.isnan(q).any()
        v = self.dropout(F.relu(self._edges_nodes_att(q, nodes)))
        assert not torch.isnan(v).any()
        return v + nodes
    def _edges_nodes_att(self, q, k):  # q:edges, k: nodes
        batchsize = q.shape[0]
        nodes = k.shape[1]
        w1 = torch.matmul(k, self.A[:self.nodefeat, :])  # [batchsize, nodes, node_feat] -> [batchsize, nodes, 1]
        w2 = torch.matmul(q, self.A[self.nodefeat:, :]).reshape(batchsize, nodes, nodes).transpose(-1,-2)  # [batchsize, nodes*nodes, node_feat] -> [batchsize, nodes*nodes, 1] -> [batchsize, nodes, nodes]
        w = torch.softmax(w1 + w2, dim=-1).transpose(-1, -2).reshape(batchsize, nodes * nodes, 1)  # [batchsize, nodes, nodes] -> [batchsize, nodes*nodes, 1]
        v = torch.mul(w, q)  # [batchsize, nodes*nodes, node_feat]
        return v.reshape(batchsize, nodes, nodes, self.nodefeat).sum(dim=1)
class FNN(nn.Module):
    def __init__(self, fea:int, dropout=0.5, ratio=4):
        super(FNN, self).__init__()
        self.inputlayer = nn.Linear(fea, ratio*fea)
        self.hidelayer = nn.Linear(ratio*fea, fea)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(fea)
    def forward(self, x):
        x = F.relu(self.inputlayer(self.layernorm(x)))
        x = self.dropout(x)
        out = F.relu(self.hidelayer(x))
        out = self.dropout(out)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        return out
class AEGAN(nn.Module):
    def __init__(self, edgefea, nodefea, dropout, head, alpha):
        super(AEGAN, self).__init__()
        self.fnn1 = FNN(edgefea)
        self.fnn2 = FNN(nodefea)
        self.nodeinteraction = NodeInteraction(edgefea, dropout)
        self.nodemultiheadgat = NodeMultiheadGAT(nodefea, nodefea, dropout, alpha, head)
        self.edgetransformer =EdgeTransformer(edgefea)
        self.nodeupdate = NodeUpdate(nodefea, edgefea, dropout)
        self.edgeupdate = EdgeUpdate(nodefea, edgefea, dropout)
        self.layernorm1 = nn.LayerNorm(edgefea)
        self.layernorm2 = nn.LayerNorm(nodefea)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, nodes, edges, adj):
        nodes = self.layernorm2(nodes)
        edges = self.layernorm1(edges)
        gate = self.nodeinteraction(edges)
        nodes, adj = self.nodemultiheadgat(nodes, adj, gate)
        edges = self.fnn1(self.edgetransformer(edges))
        nodes = self.fnn2(nodes)
        node_out = self.nodeupdate(nodes, edges)
        edge_out = self.edgeupdate(nodes, edges)
        return node_out, edge_out, adj
    
class predictor(nn.Module):
    def __init__(self, edgefea, nodefea, dropout, head, alpha, layer):
        super(predictor, self).__init__()
        self.aeganlayer = nn.ModuleList([AEGAN(edgefea, nodefea, dropout, head, alpha) for _ in range(layer)])
        # self.Maxpool = nn.MaxPool2d(kernel_size=(poolwidth, 1))
        self.fnn = nn.Sequential(
                                    nn.Linear(nodefea + edgefea, 2*(nodefea+ edgefea)),
                                    nn.PReLU(),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(2*(nodefea + edgefea), 2*(nodefea + edgefea)),
                                    nn.PReLU(),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(2*(nodefea + edgefea), 2)
                                )
    def forward(self, nodes, edges, adj):
        for m in self.aeganlayer:
            n, e, adj = m(nodes, edges, adj)
            nodes = nodes + n
            edges = edges + e
#         out = self.Maxpool(nodes).squeeze()
        out = torch.cat([nodes.mean(dim=1), edges.mean(dim=1)], dim=-1)
        out = self.fnn(out)
        return out

class Focal_loss(nn.Module):
    def __init__(self, class_num=2, gamma=2, alpha=None):
        super(Focal_loss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
            assert len(alpha) == class_num, f"expected {class_num} class, but got {len(alpha)}"
        else:
            raise ValueError(f"alpha type err:{alpha}")
            
        self.gamma = gamma
        self.class_num =  class_num
    def forward(self, pre, target, reduction='mean'):
        self.alpha = self.alpha.to(pre.device)
        pre = torch.softmax(pre, dim=-1)
#         print(pre.shape, target.shape)
        pre = pre.gather(1, target.unsqueeze(-1)).squeeze()
        log_p = torch.log(pre)
        alpha = self.alpha[target]
        loss = -alpha*torch.pow(1-pre, self.gamma)*log_p
        if reduction == 'mean':
            loss_ = loss.mean()
        elif reduction == 'sum':
            loss_ = loss.sum()
        assert not torch.isnan(loss).any(), f"pre:{pre}\ntarget:{target}\nloss:{loss}"
        assert not torch.isinf(loss).any(), f"pre:{pre}\ntarget:{target}\nloss:{loss}"
        return loss_
    
class pred_wrap(pl.LightningModule):
    def __init__(self, 
                 edgefea,
                 nodefea,
                 dropout,
                 head,
                 alpha,
                 layer,
                 weight_decay,
                 lr_rate,
                 loss_fn
                ):
        super(pred_wrap, self).__init__()
        self.predictor = predictor(edgefea, nodefea, dropout, head, alpha, layer)
        self.calculate_loss = loss_fn
        self.weight_decay = weight_decay
        self.lr_rate = lr_rate
        self.save_hyperparameters(ignore=['loss_fn'])
    def performance_compute(self, pre, tar, name):
        preidx = pre.max(dim=-1)[-1]
        # 准确率
        accuracy = (preidx == tar).sum()/tar.shape[0]
        #tp, fp, fn, tn
        tp = (preidx * tar).sum()
        fp = (preidx*(tar==0)).sum()
        fn = ((preidx==0)*tar).sum()
        tn = ((preidx==0)*(tar==0)).sum()
        precision = tp/(tp+fp)
        precision[torch.isnan(precision)] = 0
        recall = tp/(tp+fn)
        f_measure = (2*precision*recall)/(precision+recall)
        f_measure[torch.isnan(f_measure)] = 0
        mcc = (tp*tn-fp*fn)/torch.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc[torch.isnan(mcc)] = 0
        self.log_dict({f"{name}_accuracy":accuracy, f"{name}_precision":precision, \
                              f"{name}_recall":recall, f"{name}_f_measure":f_measure,f"{name}_mcc":mcc}, on_epoch=True, sync_dist=True)
    def training_step(self, batch, batch_idx):
        node, edge, adj, label = batch
        out = self.predictor(node, edge, adj)
        loss = self.calculate_loss(out, label)
        self.log("train_loss", loss.item(), sync_dist=True, on_epoch=True)
        self.performance_compute(out, label, "train")
        return loss
    def configure_optimizers(self):
        optimer = torch.optim.Adam(self.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimer,
        T_max=50,
        eta_min=1e-7
        )
        return {'optimizer': optimer, 'lr_scheduler': scheduler, "monitor": "valid_loss"}
#         return optimer
    def validation_step(self, batch, batch_idx):
        node, edge, adj, label = batch
        out = self.predictor(node, edge, adj)
        loss = self.calculate_loss(out, label)
        self.performance_compute(out, label, "valid")
        self.log("valid_loss", loss.item(), sync_dist=True)
    def test_step(self, batch, batch_idx):
        node, edge, adj, label = batch
        out = self.predictor(node, edge, adj)
        loss = self.calculate_loss(out, label)
        self.performance_compute(out, label, "test")
        self.log("test_loss", loss.item(), sync_dist=True)
        pass