# Code Transformer from scratch using Numpy👌

## Introduction

This project implements a Transformer model from scratch using only Numpy. The Transformer architecture, introduced in the paper [&#34;Attention Is All You Need&#34;](https://arxiv.org/abs/1706.03762), has revolutionized natural language processing (NLP) by replacing recurrent neural networks (RNNs) with self-attention mechanisms.

## Main Components

The Transformer model consists of the following key components:

* **Embedding Layer** : Converts input tokens into dense vector representations.
* **Positional Encoding** : Adds positional information to input embeddings.
* **Multi-Head Self-Attention** : Allows the model to focus on different parts of the input sequence simultaneously.
* **Feed-Forward Network** : Fully connected layers applied to each position independently.
* **Layer Normalization** : Stabilizes training by normalizing intermediate activations.
* **Encoder-Decoder Architecture** :
* **Encoder** : Composed of multiple identical layers that process input sequences.
* **Decoder** : Generates output sequences by attending to encoder representations.
* **Final Linear and Softmax Layers** : Produces probability distributions over vocabulary tokens.

## References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [Transformer Model Explained](https://towardsdatascience.com/transformers-141e32e69591)
* [carolinamcg (Carolina Gonçalves)](https://github.com/carolinamcg)
