# SubjectiveBiasABS
-	code for the paper "Subjective Bias in Abstractive Summarization"
-	[arxiv pdf](https://arxiv.org/pdf/2106.10084.pdf)

# introduction
- params.py：hyperparameters
- get_datasets.py：get the topk oracle sentences in article then parse
- process_dataset：turn parsed file into the format of DGL graph triplet
- model.py：the self-supervised GCN model for extractive subjective style embedding
- train.py：train, concat small graphs into a batch
- infer.py: infer the whole training set to get subjective style embedding

# detail
- negative samples of oracle sentences are uniform-sampled by the jaccard sim
