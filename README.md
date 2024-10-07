# Subjective Bias in Abstractive Summarization

<p align="center">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/dac0dcef-b5c4-4548-bb7c-d3c15868cca8">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/6aaa2840-a8da-467e-9585-c7535565aa9f">
</p>

-	code for the paper [*Subjective Bias in Abstractive Summarization*](https://arxiv.org/pdf/2106.10084.pdf)
- We examined the influence of subjective style bias in large-scale abstractive summarization datasets and introduced a Graph Convolutional Network method to capture and embed writing styles. Results demonstrate that style-clustered datasets enhance model convergence, abstraction, and generalization.

# introduction
- params.py: hyperparameters
- get_datasets.py: get the topk Oracle sentences in the article then parse
- process_dataset.py: turn parsed file into the format of DGL graph triplet
- model.py: the self-supervised GCN model for extracting subjective style embedding
- train.py: training
- infer.py: infer the whole training set to get subjective style embedding for clustering

# detail
- negative samples of Oracle sentences are uniform-sampled by the Jaccard sim
