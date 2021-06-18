import math
import torch
import torch.nn as nn
import dgl.function as fn
import params
import dgl


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 bias=True,
                 batch_normalization=False):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None

        if activation == "ReLU":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU()

        self.reset_parameters()

        if batch_normalization:
            self.bn = nn.BatchNorm1d(num_features=in_feats, affine=True)
        else:
            self.bn = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g):

        # linear
        g.ndata['h'] = torch.mm(g.ndata['h'], self.weight)

        # load node feature and forward gcn with symmetric normalization based on u_mul_e
        g.update_all(fn.u_mul_e('h', 'sym_norm', 'm'), fn.sum('m', 'h'))

        # bias
        if self.bias is not None:
            g.ndata['h'] = g.ndata['h'] + self.bias

        # batch normalization
        if self.bn:
            g.ndata['h'] = self.bn(g.ndata['h'])

        # activation
        if self.activation:
            g.ndata['h'] = self.activation(g.ndata['h'])

        return g


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_hidden_layers, activation):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()

        # input layer, no dropout
        self.layers.append(GCNLayer(in_feats, n_hidden, activation))

        # hidden layers
        for _ in range(n_hidden_layers):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation))

        # output layer, no activation
        self.layers.append(GCNLayer(n_hidden, n_hidden, None))

    def forward(self, g):
        for layer in self.layers:
            g = layer(g)
        return g


class SyntacticGraphNet(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_hidden_layers,
                 vocab_size,
                 activation="ReLU"):
        super(SyntacticGraphNet, self).__init__()
        self.gcn = GCN(in_feats, n_hidden, n_hidden_layers, activation)
        self.linear = self.Linear(in_feats, in_feats)
        self.embedding = torch.nn.Embedding(vocab_size, in_feats)

    def Linear(self, in_features, out_features, dropout=0):
        m = nn.Linear(in_features, out_features)
        nn.init.normal_(m.weight,
                        mean=0,
                        std=math.sqrt((1 - dropout) / in_features))
        nn.init.constant_(m.bias, 0)
        return nn.utils.weight_norm(m)

    def get_graph_embedding(self):
        return self.graph_embedding_summary, self.graph_embedding_pos

    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, dim=axis, keepdim=True)
        output = torch.div(input, norm)
        return output

    def forward(self, g_summary, onehot_summary, g_pos, onehot_pos, g_neg,
                onehot_neg):
        # get embedding
        h_summary = self.embedding(onehot_summary)
        h_pos = self.embedding(onehot_pos)
        h_neg = self.embedding(onehot_neg)

        # set graph feature
        g_summary.ndata['h'] = h_summary
        g_pos.ndata['h'] = h_pos
        g_neg.ndata['h'] = h_neg

        # forward gcn
        g_summary = self.gcn(g_summary)
        g_pos = self.gcn(g_pos)
        g_neg = self.gcn(g_neg)

        # get graph representation by mean pooling
        self.graph_embedding_summary = self.linear(
            dgl.mean_nodes(g_summary, 'h'))
        self.graph_embedding_pos = self.linear(dgl.mean_nodes(g_pos, 'h'))
        self.graph_embedding_neg = self.linear(dgl.mean_nodes(g_neg, 'h'))

        # normalize graph embedding
        self.graph_embedding_summary = self.l2_norm(
            self.graph_embedding_summary)
        self.graph_embedding_pos = self.l2_norm(self.graph_embedding_pos)
        self.graph_embedding_neg = self.l2_norm(self.graph_embedding_neg)

        return self.graph_embedding_summary, self.graph_embedding_pos, self.graph_embedding_neg


class SyntacticGraphScoreNet(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_hidden_layers,
                 vocab_size,
                 activation="ReLU"):
        super(SyntacticGraphScoreNet, self).__init__()
        self.gcn = GCN(in_feats, n_hidden, n_hidden_layers, activation)
        self.linear = self.Linear(2 * in_feats, 1)
        self.embedding = torch.nn.Embedding(vocab_size, in_feats)

    def Linear(self, in_features, out_features, dropout=0):
        m = nn.Linear(in_features, out_features)
        nn.init.normal_(m.weight,
                        mean=0,
                        std=math.sqrt((1 - dropout) / in_features))
        nn.init.constant_(m.bias, 0)
        return nn.utils.weight_norm(m)

    def get_graph_embedding(self):
        return self.graph_embedding_summary, self.graph_embedding_pos

    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, dim=axis, keepdim=True)
        output = torch.div(input, norm)
        return output

    def forward(self, g_summary, onehot_summary, g_pos, onehot_pos, g_neg,
                onehot_neg):
        # get embedding
        h_summary = self.embedding(onehot_summary)
        h_pos = self.embedding(onehot_pos)
        h_neg = self.embedding(onehot_neg)

        # set graph feature
        g_summary.ndata['h'] = h_summary
        g_pos.ndata['h'] = h_pos
        g_neg.ndata['h'] = h_neg

        # forward gcn
        g_summary = self.gcn(g_summary)
        g_pos = self.gcn(g_pos)
        g_neg = self.gcn(g_neg)

        # get graph representation by mean pooling
        self.graph_embedding_summary = dgl.mean_nodes(g_summary, 'h')
        self.graph_embedding_pos = dgl.mean_nodes(g_pos, 'h')
        self.graph_embedding_neg = dgl.mean_nodes(g_neg, 'h')

        # normalize graph embedding
        self.graph_embedding_summary = self.l2_norm(
            self.graph_embedding_summary)
        self.graph_embedding_pos = self.l2_norm(self.graph_embedding_pos)
        self.graph_embedding_neg = self.l2_norm(self.graph_embedding_neg)

        # score the relation using linear
        score_sum_pos = torch.sigmoid(
            self.linear(
                torch.cat(
                    (self.graph_embedding_summary, self.graph_embedding_pos),
                    dim=1)))
        score_sum_neg = torch.sigmoid(
            self.linear(
                torch.cat(
                    (self.graph_embedding_summary, self.graph_embedding_neg),
                    dim=1)))
        score_sum_pos = torch.mean(score_sum_pos)
        score_sum_neg = torch.mean(score_sum_neg)

        return score_sum_pos, score_sum_neg
