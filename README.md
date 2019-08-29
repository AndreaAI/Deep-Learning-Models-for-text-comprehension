# Deep Learning Models for text comprehension

This repository contains information and explanation about Neural Networks and some parts of my Master Thesis "Development of Deep Learning Models for text comprehension using NLP techniques", as well as the implemented code.


## Artificial Neural Networks

Artificial neural networks (ANN) are mathematical algorithms rising up from the idea of imitating biological neural networks behaviour,
particularly human ones.

Following this human brain simulation, these networks consist of a set of artificial neurons (nodes) distributed in different interconnected layers with some specific configuration. A network counts with at least two layers, an input one and an output one, and some hidden layers can be also present, and they will affect the configuration complexity. Every connection between two neurons has a weight associated, which indicates the importance of that connection, and it will be used to weight the inputs of the corresponding neuron, together with a bias. These weights will need to be estimated from the dataset; this is called the learning process or training of the neural network.

Neural networks, in terms of configuration, can be feedforward (without feedback) or recurrent netwoks, depeding on if the output of a neuron can only be the input of neurons in the next layer or, on the contrary, the output of a neuron can also be the input of neurons in previous layers or in the current one.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/nn.PNG" width="700" height="420">
</p>

The idea behind these networks is being able to count with a kind of memory where previous information could be stored, as well as being able to access it multiple times and reasoning over it to imitate more accurately a biological neural network. This recurrence can be interpreted as the existence of multiple layers in the network, where the information is passed on from layer to layer, as it can be seen in the following figure:

![Recurrent neural network schema](/images/rnn4.png)


### Backpropagation and gradient descendent

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/deeprnn.png" width="700" height="450">
</p>

### Long Short term Memory Networks (LSTM)
<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/LSTM2.png" width="450" height="320">
</p>

### Gated Recurrent Units (GRU)
<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/GRU.png" width="450" height="350">
</p>

## Memory Networks

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2.png" width="550" height="220">
</p>

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memnn21.png" width="550" height="320">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memnn22.png" width="600" height="230">
</p>

### End-to-End Memory Networks (MemN2N)

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2n21.png" width="550" height="300">
</p>

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2n22.png" width="700" height="300">
</p>

### Key-Value Memory Networks (KV-MemNN)

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/kvmemnn2.png" width="700" height="580">
</p>
