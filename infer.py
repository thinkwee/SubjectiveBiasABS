import torch
import _pickle as pickle
from torch.utils.data import DataLoader
import params
from argparse import ArgumentParser
from model import SyntacticGraphNet, SyntacticGraphScoreNet
import dgl
from tqdm import tqdm


def collate(samples):
    graph_summary, onehot_summary, graph_pos, onehot_pos, graph_neg, onehot_neg = map(
        list, zip(*samples))

    batched_graph_summary = dgl.batch(graph_summary)
    batched_graph_pos = dgl.batch(graph_pos)
    batched_graph_neg = dgl.batch(graph_neg)

    onehot_summary = sum(onehot_summary, [])
    onehot_pos = sum(onehot_pos, [])
    onehot_neg = sum(onehot_neg, [])

    return batched_graph_summary, torch.tensor(
        onehot_summary), batched_graph_pos, torch.tensor(
            onehot_pos), batched_graph_neg, torch.tensor(onehot_neg)


if __name__ == "__main__":
    # parse argument
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        help="dataset name, small|middle|large_undirected|large_directed",
        default="middle")
    parser.add_argument("-m",
                        "--model",
                        help="model name, embedding|score",
                        default="score")

    args = parser.parse_args()
    dataset_path = "./data/" + args.data + ".bin"
    model_name = args.model
    save_name = 'model_' + model_name + "_" + args.data

    # load vocab
    type2id = pickle.load(open("./data/type2id", "rb"))

    # loss function
    loss_func = torch.nn.TripletMarginLoss(margin=params.loss_margin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    if model_name == "embedding":
        model = SyntacticGraphNet(in_feats=params.hidden_size,
                                  n_hidden=params.hidden_size,
                                  n_hidden_layers=1,
                                  vocab_size=len(type2id)).to(device)
    elif model_name == "score":
        model = SyntacticGraphScoreNet(in_feats=params.hidden_size,
                                       n_hidden=params.hidden_size,
                                       n_hidden_layers=1,
                                       vocab_size=len(type2id)).to(device)

    # load model
    model.load_state_dict(torch.load('./save_model/' + save_name + '.pkl'))
    model.eval()

    # infer
    with torch.no_grad():
        for epoch in range(params.bin_total):
            # load data bin
            train_bin = pickle.load(open(dataset_path + str(epoch), "rb"))
            data_loader = DataLoader(train_bin,
                                     batch_size=params.batch_size_infer,
                                     shuffle=False,
                                     collate_fn=collate,
                                     num_workers=4)

            # infer
            for iter, (gs, os, gp, op, gn, on) in tqdm(enumerate(data_loader)):

                gs.to(device)
                os = os.to(device)
                gp.to(device)
                op = op.to(device)
                gn.to(device)
                on = on.to(device)

                _ = model(gs, os, gp, op, gn, on)

                graph_embedding_summary, graph_embedding_pos = model.get_graph_embedding(
                )

                concatenated = torch.cat(
                    [graph_embedding_summary, graph_embedding_pos], 1)

                if iter == 0 and epoch == 0:
                    result = concatenated
                else:
                    result = torch.cat([result, concatenated], 0)
            del (train_bin)
            del (data_loader)

    print(result.size())
    pickle.dump(result, open("./clustering/" + save_name + ".pkl", "wb"))
