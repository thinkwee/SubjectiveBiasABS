import torch
import _pickle as pickle
import dgl
from torch.utils.data import DataLoader
import params
import torch.optim as optim
from model import SyntacticGraphNet, SyntacticGraphScoreNet
# from matplotlib import pyplot as plt
from tqdm import tqdm
from visdom import Visdom
from argparse import ArgumentParser


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


def model_summary(model):
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def margin_triplet_score_loss(score_pos, score_neg, margin):
    return max(0, score_neg - score_pos + margin)


def load_data(dataset_path, bin_num):
    # load data
    train_bin = pickle.load(open(dataset_path + str(bin_num), "rb"))
    viz.text(dataset_path + str(bin_num) + " loaded", win='log', append=True)
    data_loader = DataLoader(train_bin,
                             batch_size=params.batch_size,
                             shuffle=True,
                             collate_fn=collate)
    return train_bin, data_loader


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

    # visualize loss
    viz = Visdom(env=save_name)
    opts_loss = {
        'title': save_name,
        'xlabel': 'every batch',
        'ylabel': 'Loss',
        'showlegend': 'true'
    }
    opts_dis_sim = {
        'title': 'Similarity Distance',
        'xlabel': 'every batch',
        'ylabel': 'Distance',
        'showlegend': 'true'
    }
    opts_dis_score = {
        'title': 'Pos/Neg Diff Score',
        'xlabel': 'every batch',
        'ylabel': 'Score Diff',
        'showlegend': 'true'
    }
    opts_dis_embed = {
        'title': 'Embedding Distance',
        'xlabel': 'every batch',
        'ylabel': 'Distance',
        'showlegend': 'true'
    }

    type2id = pickle.load(open("./data/type2id", "rb"))
    viz.text("vocab loaded", win='log', append=False)

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

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    step = 0
    loss_save = []
    dis_sim_save = []
    dis_embed_save = []

    if model_name == "embedding":
        for epoch in range(params.epoches):
            bin_num = epoch % params.bin_total
            if epoch % 20 == 0:
                train_bin, data_loader = load_data(dataset_path, bin_num)
            for iter, (gs, os, gp, op, gn, on) in tqdm(enumerate(data_loader)):
                update = 'append' if step > 1 else None

                gs.to(device)
                os = os.to(device)
                gp.to(device)
                op = op.to(device)
                gn.to(device)
                on = on.to(device)

                graph_embedding_summary, graph_embedding_pos, graph_embedding_neg = model(
                    gs, os, gp, op, gn, on)

                loss = loss_func(graph_embedding_summary, graph_embedding_pos,
                                 graph_embedding_neg)

                optimizer.zero_grad()
                try:
                    loss.backward()
                except AttributeError:
                    print(loss)
                    continue
                optimizer.step()

                step += 1

                loss_value = loss.detach().item()

                sim_sum_p = torch.norm(graph_embedding_summary -
                                       graph_embedding_pos,
                                       dim=1,
                                       out=None,
                                       keepdim=False)
                sim_sum_n = torch.norm(graph_embedding_summary -
                                       graph_embedding_neg,
                                       dim=1,
                                       out=None,
                                       keepdim=False)

                dis_sim = torch.mean(sim_sum_n - sim_sum_p,
                                     dim=0).detach().item()

                raw_embedding_summary, raw_embedding_pos = model.get_graph_embedding(
                )
                dis_embedding = torch.norm(raw_embedding_summary -
                                           raw_embedding_pos,
                                           dim=1,
                                           out=None,
                                           keepdim=False)
                dis_embedding_mean = torch.mean(dis_embedding,
                                                dim=0).detach().item()

                loss_save.append(loss_value)
                dis_sim_save.append(dis_sim)
                dis_embed_save.append(dis_embedding_mean)

                if step % params.print_every == 0:
                    viz.text('step {}, loss {:.4f}'.format(
                        step,
                        loss.detach().item()),
                             win='log',
                             append=True)
                    viz.line(X=torch.FloatTensor([step]),
                             Y=torch.FloatTensor([loss_value]),
                             win='loss',
                             update=update,
                             opts=opts_loss,
                             name='train')

                    viz.line(X=torch.FloatTensor([step]),
                             Y=torch.FloatTensor([dis_sim]),
                             win='dis_sim',
                             update=update,
                             opts=opts_dis_sim,
                             name='train')

                    viz.line(X=torch.FloatTensor([step]),
                             Y=torch.FloatTensor([dis_embedding_mean]),
                             win='dis_embedding',
                             update=update,
                             opts=opts_dis_embed,
                             name='train')
    elif model_name == "score":
        for epoch in range(params.epoches):
            bin_num = epoch % params.bin_total
            if epoch % 20 == 0:
                train_bin, data_loader = load_data(dataset_path, bin_num)
            for iter, (gs, os, gp, op, gn, on) in tqdm(enumerate(data_loader)):
                update = 'append' if step > 1 else None

                gs.to(device)
                os = os.to(device)
                gp.to(device)
                op = op.to(device)
                gn.to(device)
                on = on.to(device)

                score_sum_pos, score_sum_neg = model(gs, os, gp, op, gn, on)

                loss = margin_triplet_score_loss(score_sum_pos, score_sum_neg,
                                                 params.loss_margin_score)

                optimizer.zero_grad()
                try:
                    loss.backward()
                except AttributeError:
                    print(loss)
                    continue
                optimizer.step()

                step += 1

                loss_value = loss.detach().item()

                dis_score = score_sum_pos.detach().item(
                ) - score_sum_neg.detach().item()

                raw_embedding_summary, raw_embedding_pos = model.get_graph_embedding(
                )
                dis_embedding = torch.norm(raw_embedding_summary -
                                           raw_embedding_pos,
                                           dim=1,
                                           out=None,
                                           keepdim=False)
                dis_embedding_mean = torch.mean(dis_embedding,
                                                dim=0).detach().item()

                if step % params.print_every == 0:
                    viz.text('step {}, loss {:.4f}'.format(
                        step,
                        loss.detach().item()),
                             win='log',
                             append=True)
                    viz.line(X=torch.FloatTensor([step]),
                             Y=torch.FloatTensor([loss_value]),
                             win='loss',
                             update=update,
                             opts=opts_loss,
                             name='train')

                    viz.line(X=torch.FloatTensor([step]),
                             Y=torch.FloatTensor([dis_score]),
                             win='dis_score',
                             update=update,
                             opts=opts_dis_score,
                             name='train')

                    viz.line(X=torch.FloatTensor([step]),
                             Y=torch.FloatTensor([dis_embedding_mean]),
                             win='dis_embedding',
                             update=update,
                             opts=opts_dis_embed,
                             name='train')

                    loss_save.append(loss_value)
                    dis_sim_save.append(dis_score)
                    dis_embed_save.append(dis_embedding_mean)

    torch.save(model.state_dict(), './save_model/' + save_name + '.pkl')
    pickle.dump(loss_save, open("./record/loss_" + save_name + '.pkl', "wb"))
    pickle.dump(dis_sim_save,
                open("./record/dis_sim_" + save_name + '.pkl', "wb"))
    pickle.dump(dis_embed_save,
                open("./record/dis_embed_" + save_name + '.pkl', "wb"))
