# A Simple Framework for Contrastive Learning of Visual Representations

## Objective

Yet another implementation of the [SimCLR paper](https://www.arxiv.org/abs/2002.05709), on which
I heavily rely for my master's thesis.
The other implementations either didn't lend themselves for easy use with my datasets/other code, or I encountered installation/performance issues, so I decided to 
write my own.
Also, it's a good excercise to implement papers.

**Note**: Since I'm training on imbalanced datasets, I included some code in `train.py` to measure the euclidean and cosine distances between positive views per class, and plot them. If this is a problem, feel free to comment out all code related to measuring or plotting these values.

## Structure

The `train.py` file trains the SimCLR model with no labels using the contrastive loss as described in the paper.

`train_lincls.py` trains a linear classifier on top of the frozen pre-trained SimCLR model to evaluate the representations. The default is to train a linear
classifier layer for 100 epochs to measure the representation quality.

To see the options, run
`python train.py -h`

The performance after training a linear classifier is given below.
**Note** that SSL models are usually pre-trained for much longer, these results
are just to verify against the paper's results (Figure B.7)

Architecture | Encoder Output Dim. | Projector Otput Dim | Training Epochs | Acc.
---|---|---|---|---|
resnet18 | 512 | 128 | 100 | 79.838

```bibtex
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
```

