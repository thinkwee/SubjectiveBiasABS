import dgl
import _pickle as pickle
import torch
import params
import networkx as nx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from argparse import ArgumentParser


def show_statistic(g):
    print("nodes: %d" % (g.number_of_nodes()))
    print("edges: %d" % (g.number_of_edges()))
    print(g.ndata['norm'])
    return


def create_type2id(dependency2id, pos2id):
    '''combine dependecy2id and pos2id to type2id 
    '''

    type2id = dict()
    count = 0
    for dep in dependency2id:
        type2id[dep] = count
        count += 1

    for pos in pos2id:
        type2id[pos] = count
        count += 1

    return type2id


def extract_bin(atom,
                sth2id,
                graph,
                name,
                heterogeneous=False,
                visualize=False,
                save_gml=False):
    ''' parse data in each atom of train/valid/test.bin and create dgl_graph
    each sample contains topk + 1 atom which are[summary,pos,neg_1,...,neg_topk-1]
    each atom is a dictionary : {'sentence':str,'edges':list of tuple(src_word,src_id, src_pos,dep,tgt_word,tgt_id, tgt_pos)}
    '''
    if heterogeneous:
        return None
    else:
        g = dgl.DGLGraph()

        # we treat dep as a node but different dependency relations with same dep type are treated as one node
        word_count = len(atom['edges'])
        dep_set = set([x[3] for x in atom['edges']])
        dep_count = len(dep_set)
        g.add_nodes(word_count + dep_count)
        onehot = [-1 for _ in range(word_count + dep_count)]

        # transform word and dep into nodes id
        # add dep node onehot
        dep2node = dict()
        count = word_count
        for dep in dep_set:
            dep2node[dep] = count
            onehot[count] = sth2id[dep]
            count += 1

        # add edges and word node onehot
        edge_list = []
        if count == word_count + dep_count:
            for idx, edge in enumerate(atom['edges']):
                # add src -> dep
                edge_list.append(tuple([edge[1], dep2node[edge[3]]]))
                # add dep -> tgt
                edge_list.append(tuple([dep2node[edge[3]], edge[5]]))
                # add attribute (word pos)
                onehot[idx] = sth2id[edge[2]]
        else:
            print("wrong")
            print(count, word_count, dep_count)
            print(atom['sentence'])

        # save gml graph for better visualization
        if save_gml:
            print(atom['index'])
            print(atom['sentence'])
            if atom['index'] == 9:
                G_gml = nx.Graph()
                for idx, edge in enumerate(atom['edges']):
                    # add src -> dep
                    G_gml.add_edge(edge[2] + '_' + str(edge[1]), edge[3])
                    # add dep -> tgt
                    G_gml.add_edge(edge[3], edge[6] + '_' + str(edge[5]))
                nx.write_gml(G_gml,
                             './G_' + name + "_" + str(atom['index']) + ".gml")

        # add edges into DGL graph
        # double direction?
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        if graph == "undirected":
            g.add_edges(dst, src)

        # add norm for all nodes
        unnormed = g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())
        g.ndata['norm'] = torch.sqrt(unnormed.float())

        # add symmetric norm value on edge
        for i in range(g.number_of_edges()):
            src, tgt = g.find_edges(i)
            g.edges[i].data['sym_norm'] = 1.0 / \
                (g.nodes[src].data['norm'] * g.nodes[tgt].data['norm'])

        if visualize:
            # visualize graph
            id2sth = {v: k for k, v in sth2id.items()}
            labels = dict()
            for idx, i in enumerate(onehot):
                labels[idx] = id2sth[i]

            nx_G = g.to_networkx()
            # pos = nx.kamada_kawai_layout(nx_G)
            pos = nx.nx_agraph.graphviz_layout(nx_G, prog='dot')
            nx.draw(nx_G,
                    pos,
                    with_labels=True,
                    labels=labels,
                    node_size=800,
                    node_color=[[.7, .7, .7]],
                    arrowsize=5)
            plt.show()

        return g, onehot


def build_homogeneous(train_bin, graph):
    ''' only use pos and dependency as feature of nodes, all nodes share the same type
    '''
    # get id
    dependency2id = pickle.load(open("./data/dependency2id", "rb"))
    pos2id = pickle.load(open("./data/pos2id", "rb"))
    type2id = create_type2id(dependency2id, pos2id)
    pickle.dump(type2id, open("./data/type2id", "wb"))

    # prepare processed sample lists
    result_list = []
    sentence_pair_list = []
    id_list = []

    count = 0
    bin_num = 0

    if graph == "directed":
        prefix = "./data/large_directed"
    elif graph == "undirected":
        prefix = "./data/large_undirected"

    for sample in tqdm(train_bin):
        try:
            summary_graph, summary_onehot = extract_bin(sample[0],
                                                        type2id,
                                                        graph,
                                                        "gold",
                                                        save_gml=True)
        except IndexError:
            print(sample)
            exit

        s = input()

        pos_graph, pos_onehot = extract_bin(sample[1],
                                            type2id,
                                            graph,
                                            "pos",
                                            save_gml=True)

        rand_choose = random.randint(2, params.topk)
        neg_graph, neg_onehot = extract_bin(sample[rand_choose],
                                            type2id,
                                            graph,
                                            "neg",
                                            save_gml=True)

        temp = tuple([
            summary_graph, summary_onehot, pos_graph, pos_onehot, neg_graph,
            neg_onehot
        ])
        result_list.append(temp)
        sentence_pair_list.append(
            tuple([sample[0]['sentence'], sample[1]['sentence']]))
        id_list.append(sample[0]['index'])

        count += 1

        if count % params.bin_size == 0:
            pickle.dump(result_list, open(prefix + ".bin" + str(bin_num),
                                          "wb"))
            pickle.dump(
                sentence_pair_list,
                open(prefix + "sentence_pair" + ".bin" + str(bin_num), "wb"))
            pickle.dump(id_list,
                        open(prefix + "id_list" + ".bin" + str(bin_num), "wb"))
            del (result_list)
            del (sentence_pair_list)
            del (id_list)
            result_list = []
            sentence_pair_list = []
            id_list = []
            bin_num += 1


if __name__ == "__main__":
    # parse argument
    parser = ArgumentParser()
    parser.add_argument("-g",
                        "--graph",
                        help="graph type, undirected|directed",
                        default="undirected")

    args = parser.parse_args()
    graph = args.graph

    train_bin = pickle.load(open("./data/train.bin", "rb"))
    print("train_bin loaded")
    build_homogeneous(train_bin, graph)
