import spacy
from tqdm import tqdm
import _pickle as pickle
import params


class SummaryDataProcessor():
    def __init__(self, topk):
        self.nlp = spacy.load("en_core_web_lg")
        self.doc = ""
        self.summary = ""
        self.topk = topk
        self.doc_sentences = []
        self.summary_sentences = []
        self.candidate = []
        self.pos2id = dict()
        self.dependency2id = dict()
        self.word2id = dict()
        self.word_count = 0
        self.dependency_count = 0
        self.pos_count = 0

    def update(self, doc, summary):
        if "\n" in doc:
            self.doc = doc.split("\n")[0]
        else:
            self.doc = doc

        if "\n" in summary:
            self.summary = summary.split("\n")[0]
        else:
            self.summary = summary

        self.doc_sentences = self.doc.split(" . ")[:-1]

        if " . " in self.summary:
            self.summary_sentences = self.summary.split(" . ")[:-1]
        else:
            self.summary_sentences = [self.summary]

        self.candidate = []

    def jacsim(self, str1, str2):
        list1 = str1.split(" ")
        list2 = str2.split(" ")
        unionlength = float(len(set(list1) | set(list2)))
        interlength = float(len(set(list1) & set(list2)))
        return float('%.3f' % (interlength / unionlength))

    def make_k_candidate(self):
        for summary_sentence in self.summary_sentences:
            jacsim_list = [
                self.jacsim(sentence, summary_sentence)
                for sentence in self.doc_sentences
            ]
            sorted_sentences = [
                item[1]
                for item in sorted(zip(jacsim_list, self.doc_sentences),
                                   reverse=True)
            ]
            pos_sample = sorted_sentences[0]
            step = (len(jacsim_list) - 1) // (self.topk - 1)
            try:
                temp = sorted_sentences[1::step]  # negative sample set
            except ValueError:
                print(step)
                print(len(self.doc_sentences))
                print(len(jacsim_list))
                print(self.doc_sentences)
                print(self.doc)
                break
            temp = temp[:self.topk - 1]  # only topk-1 negative sample
            temp.insert(0, pos_sample)  # second is pos_sample
            temp.insert(0, summary_sentence)  # first is summary
            self.candidate.append(temp)

            break

    def build_graph(self, index):
        sample = []
        for summary_wise in self.candidate:
            for sentence in summary_wise:
                sample_dict = dict()
                doc = self.nlp(sentence)
                sample_dict['sentence'] = sentence
                sample_dict['index'] = index
                edges = []
                for token in doc:
                    if token.text not in self.word2id:
                        self.word2id[token.text] = self.word_count
                        self.word_count += 1
                    if token.pos_ not in self.pos2id:
                        self.pos2id[token.pos_] = self.pos_count
                        self.pos_count += 1
                    if token.dep_ not in self.dependency2id:
                        self.dependency2id[token.dep_] = self.dependency_count
                        self.dependency_count += 1
                    edges.append(
                        tuple([
                            token.text, token.i, token.pos_, token.dep_,
                            token.head.text, token.head.i, token.head.pos_
                        ]))
                sample_dict['edges'] = edges
                sample.append(sample_dict)
        return sample


if __name__ == "__main__":
    sdp = SummaryDataProcessor(params.topk)
    src = []
    tgt = []

    max_count = params.train_count
    result = []

    with open("./data/cnndm/train.src", "r") as f:
        for idx, line in enumerate(f):
            src.append(line)
            if idx == max_count:
                break

    with open("./data/cnndm/train.tgt", "r") as f:
        for idx, line in enumerate(f):
            tgt.append(line)
            if idx == max_count:
                break

    save_every = params.train_count // 10

    skip_count = 0
    for i in tqdm(range(max_count)):
        sdp.update(src[i], tgt[i])
        if len(sdp.doc_sentences) < params.topk:
            skip_count += 1
            continue
        sdp.make_k_candidate()
        graph = sdp.build_graph(i)
        if len(graph) == 0:
            print(sdp.candidate)
            print(sdp.doc_sentences)
            print(sdp.summary_sentences)
            print([sdp.doc, sdp.summary])
        else:
            result.append(graph)
        if i % save_every == 0:
            pickle.dump(result, open("./data/train.bin", "wb"))
            pickle.dump(sdp.word2id, open("./data/word2id", "wb"))
            pickle.dump(sdp.dependency2id, open("./data/dependency2id", "wb"))
            pickle.dump(sdp.pos2id, open("./data/pos2id", "wb"))

    pickle.dump(result, open("./data/train.bin", "wb"))
    pickle.dump(sdp.word2id, open("./data/word2id", "wb"))
    pickle.dump(sdp.dependency2id, open("./data/dependency2id", "wb"))
    pickle.dump(sdp.pos2id, open("./data/pos2id", "wb"))
    print("skipped %d samples" % skip_count)
