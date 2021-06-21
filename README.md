# SubjectiveBiasABS
-	code for the paper "Subjective Bias in Abstractive Summarization"
-	[arxiv pdf](https://arxiv.org/pdf/2106.10084.pdf)

# introduction
- params.py：hyperparameters
- get_datasets.py：get the topk oracle sentences in the article then parse
- process_dataset.py：turn parsed file into the format of DGL graph triplet
- model.py：the self-supervised GCN model for extracting subjective style embedding
- train.py：training
- infer.py: infer the whole training set to get subjective style embedding for clustering

# detail
- negative samples of oracle sentences are uniform-sampled by the jaccard sim
