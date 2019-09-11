# Deep Learning Models for text comprehension

This repository contains information and explanation about Neural Networks and some parts of my Master Thesis "Development of Deep Learning Models for text comprehension using NLP techniques", as well as the implemented code.


## Artificial Neural Networks

Artificial neural networks (ANN) are mathematical algorithms rising up from the idea of imitating biological neural networks behaviour,
particularly human ones.

Following this human brain simulation, these networks consist of a set of artificial neurons (nodes) distributed in different interconnected layers with some specific configuration. A network counts with at least two layers, an input one and an output one, and some hidden layers can be also present, and they will affect the configuration complexity. Every connection between two neurons has an associated weight, which indicates the importance of that connection, and it will be used to weight the inputs of the corresponding neuron, together with a bias. These weights will need to be estimated from the dataset; this is called the learning process or training of the neural network.

Neural networks, in terms of configuration, can be feedforward (without feedback) or recurrent networks, depending on if the output of a neuron can only be the input of neurons in the next layer or, on the contrary, the output of a neuron can also be the input of neurons in previous layers or in the current one.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/nn.PNG" width="700" height="420">
</p>

The idea behind these networks is being able to count with a kind of memory where previous information could be stored, as well as being able to access it multiple times and reasoning over it to imitate more accurately a biological neural network. This recurrence can be interpreted as the existence of multiple layers in the network, where the information is passed on from layer to layer, as it can be seen in the following figure:

![Recurrent neural network schema](/images/rnn4.png)

Without loss of generality, assume a recurrent neural network has only three layers: input, hidden and output,
and that its hidden layer is connected to itself, apart from to the output layer. Let $n$, $m$ and $l$ be the number of 
existing neurons in the input, hidden and output layer, respectively. Consider $t$ as the indicator of the observation times. Defining

\begin{itemize}
\item[$\triangleright$] $x_{t}^{i}$ the inputs of the network at instant $t$,  $i=1,\dots,n$
\item[$\triangleright$] $h_{t}^{j}$ the outputs of them hidden layer at instant $t$,  $j=1,\dots,m$ 
\item[$\triangleright$] $y_{t}^{k}$ the final outputs of the network at instant $t$, $k=1,\dots,l$
\item[$\triangleright$] $f$ and $g$ the exit function of the hidden and output layer, respectively
\item[$\triangleright$] $w^{1}_{ij}$ the weights associated to the connections between the neurons in the input layer and in the hidden one
\item[$\triangleright$] $w^{2}_{jh}$ the weights associated to the connections between the neurons in the hidden layer
\item[$\triangleright$] $w^{3}_{jk}$ the weights associated to the connections between the neurons in the hidden layer and in the output one
\item[$\triangleright$] $\theta^{1}_{j}$ y $\theta^{2}_{k}$ the bias associated to neuron $j$ from the hidden layer and neuron $k$ from the output layer
\end{itemize}

A recurrent neural network can be defined by the following expression:

$$ h_{t}^{j}=f\bigg(\sum_{i=1}^{n}w^{1}_{ij}x_{t}^{i} + \sum_{h=1}^{m}w^{2}_{jh}h_{t-1}^{h} + \theta^{1}_{j}\bigg),\hskip0.5cm y_{t}^{k} = g \bigg( \sum_{j=1}^{m}w^{3}_{jk}h_{t}^{j} + \theta^{2}_{k}\bigg).$$


From now on, the following matrix notation will be used in order to simplify the formulas defining the networks. This representation can be seen as considering that there is only one node per each layer.

\begin{itemize}
\item[$\triangleright$]  $x_{t}=(x_{t}^{1},\dots,x_{t}^{n})$
\item[$\triangleright$] $h_{t}=(h_{t}^{1},\dots,h_{t}^{m})$
\item[$\triangleright$] $y_{t}=(y_{t}^{1},\dots,y_{t}^{l})$
\item[$\triangleright$] $W^{1}=(w^{1}_{ij})$ the matrix of the weights associated to the connections between the neurons in the input layer and in the hidden one
\item[$\triangleright$] $W^{2}=(w^{2}_{jh})$ the matrix of the weights associated to the connections between the neurons in the hidden layer
\item[$\triangleright$] $W^{3}=(w^{3}_{jk})$ the matrix of the weights associated to the connections between the neurons in the hidden layer and in the output one
\item[$\triangleright$] $\theta^{1}=(\theta^{1}_{j})$ y $\theta^{2}=(\theta^{2}_{k})$ the matrixes (vectors) of the bias associated to neuron $j$ from the hidden layer and neuron $k$ from the output layer
\end{itemize}


So, considering this notation, the recurrent neural network can be defined by the following expression:

$$ h_{t}=f(W^{1}x_{t} + W^{2}h_{t-1} + \theta^{1}),\hskip0.5cm y_{t} = g (W^{3}h_{t} + \theta^{2}).$$


If several hidden layers exist, a layer will receive as input the initial inputs, in addition to the outputs from the previous layer,
and the final output layer will receive as inputs the outputs from all hidden layers. This makes the training of deep networks easier due to the number of processing steps from the bottom to the top of the network, and also it allows to soften the gradient vanishing problem.
A representation of a network with several hidden layers can be seen in the next picture:


<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/deeprnn.png" width="700" height="450">
</p>


### Long Short term Memory Networks (LSTM)

LSTM are recurrent neural networks whose neurons in the hidden layer are replaced by memory cells. These cells have a 'gates system' that decides what information should be stored in the memory, what information should be forgotten and what should be transferred to the rest of the layers. Typically, the activation functions considered in this kind of networks are the sigmoid function and the hyperbolic tangent. The structure of LSTM hidden layers can be represented, for a time instant $t$, as follows:

\begin{table}[htbp]
	\begin{center}
		\begin{tabular}{ll}
			$m_{t}=f_{t}\ast m_{t-1}+g_{t}\ast i_{t}$, & inner memory \\
			&\\
			$i_{t}=\sigma(W_{1}^{i}x_{t}+W_{2}^{i}h_{t-1})$, & input gate\\
			& \\
			$f_{t}=\sigma(W_{1}^{f}x_{t}+W_{2}^{f}h_{t-1})$, & forget gate\\
			& \\
			$o_{t}=\sigma(W_{1}^{o}x_{t}+W_{2}^{o}h_{t-1})$, & output gate\\			
			& \\
			$g_{t}=\varsigma(W_{1}^{g}x_{t}+W_{2}^{g}h_{t-1})$, & input hidden state\\
			& \\
			$h_{t}=\varsigma(m_{t})\ast o_{t}$, & output hidden state\\						
		\end{tabular}
	\end{center}
\end{table}

where $x_{t}$ is the vector representing the entrance to the corresponding layer at instant $t$, $\sigma$ represents the sigmoid function, $\varsigma$ the hyperbolic tangent function, and $W_{1}$, $W_{2}$ are the weight matrixes that we need to estimate. In each instant of time, based on $x_{t}$, the LSTM generates the corresponding $h_{t}$ y $m_{t}$.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/LSTM2.png" width="450" height="320">
</p>

### Gated Recurrent Units (GRU)

Gated Recurrent Units, GRU, have less parameters than LSTM. They combine the input and forget gates into a unique one called update gate, and also the inner memory with the hidden state. Its structure is defined by the following expressions:

\begin{table}[h!]
	\begin{center}
		\begin{tabular}{ll}
			$z_{t}=\sigma(W_{1}^{z}x_{t}+W_{2}^{z}h_{t-1})$, & update gate\\
			& \\
			$r_{t}=\sigma(W_{1}^{r}x_{t}+W_{2}^{r}h_{t-1})$, & reset gate\\
			& \\
			$h_{t}=(1-z_{t})\ast h_{t-1}+z_{t} \ast\varsigma(W_{1}^{h}x_{t}+W_{2}^{h}(r_{t}\ast h_{t-1}))$, & output hidden state\\						
		\end{tabular}
	\end{center}
\end{table}


where $x_{t}$ is the vector representing the entrance to the corresponding layer at instant $t$, $\sigma$ represents the sigmoid function, $\varsigma$ the hyperbolic tangent function, and $W_{1}$, $W_{2}$ are the weight matrixes that we need to estimate. In each instant of time, based on $x_{t}$, the GRU generates the corresponding $h_{t}$.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/GRU.png" width="450" height="350">
</p>

## Memory Networks

Memory networks, as the name implies, consist of a memory $\textbf{m}$ (an array of objects named as $\textbf{m}_{i}$) on which we can read and write information, and four components. The input component, $I$, is the one in charge of encoding the input $x$ into the corresponding internal representation, $I(x)$. The generalization component, $G$, updates every memory $m_{i}$ with the new input: $m_{i}=G(m_{i}, I(x), \textbf{m})$, $\forall i=1,\dots,N$. The output component, $O$, is in charge of processing an output $o$ from the new input $I(x)$ and the memory $\textbf{m}$, $o=O(I(x),\textup{\textbf{m}})$. Finally, the response component $R$ decodes the output $o$ providing an interpretable answer $r$, like a text or an action: $r=R(o)$. In any of these components, any machine learning model could be used, but in this case we will focus on having a recurrent neural network in some of the components, thus obtaining a memory network.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2.png" width="550" height="220">
</p>

In the basic model, $I$ gets an input $x$ that can be encoded into an internal representation before moving forward to the component $G$, and this component will store it in the next available memory slot. The neural network is introduced in the component $O$, that in the case of question answering, will be the one in charge of finding the $k$ supporting sentences that answer the matter in question, being $k\leq N$ and $N$ the number of memory slots. $R$ will generate a response from the input $x$ (a question in this case) and the selected supporting memories. For $k$ recurrences, the schema of a memory neural network can be represented as follows:


<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memnn21.png" width="550" height="320">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memnn22.png" width="600" height="230">
</p>


### End-to-End Memory Networks (MemN2N)

This network follows the same structure as the one already introduced, but it is trained end-to-end. This training strategy consists of training the whole system at the same time, from the input modules to the output ones. This implies that the network needs less supervision during the training phase, done by backpropagation, so it can be applied in more realistic scenarios. In this model, the recurrence can be found in the multiple accesses to the memory before generating an output.

We will explain how it works when applying it to find answers about a specific question in a text:

Consider $x_{1},\dots,x_{n}$ the set of input data that need to be stored in the memory, $q$ the question and $a$ its answer. The model consists of three phases:

\begin{enumerate}
	\item Input data $\{x_{i}\}$ is encoded into memory vectors $\{m_{i}\}$ of dimension $d$, by a matrix $A$, and the question $q$ is encoded by a matrix $B$ into the vector $u$. Next, a probability $p_{i}$ is assigned to each memory $m_{i}$ when comparing the question to the input:
	$$m_{i}=Ax_{i}, \hskip0.5cm u=Bq, \hskip0.5cm  p_{i}=Softmax(u^{T}m_{i}). $$
	\item Equally, the inputs $x_{i}$ are encoded as output vectors $c_{i}$ by a matrix $C$, and the system returns the vector of the weighted sum of the outputs $c_{i}$ by their associated probabilities:	
	$$ c_{i}=Cx_{i}, \hskip0.5cm o=\sum_{i}p_{i}c_{i}. $$		
	\item Lastly, the final prediction $\hat{a}$ is generated by the sum of the vector $o$ and the encoded question $u$, multiplied by a matrix of weights $W$ and applying later on the function\textit{Softmax}: 	
	$$ \hat{a}=Softmax(W(o+u)).$$	
\end{enumerate}


The model learns the matrixes $A,B,C,W$ by minimizing the standard cross entropy loss between $\hat{a}$ and the real answer $a$.




<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2n21.png" width="550" height="300">
</p>



#### Modifications
If we want the model to make multiple hops, it needs to be built with multiple layers instead of just one. This way the model will provide better results when predicting the answers, since it is able to access the memory multiple times. How it works is the same, but now we will have different matrixes $A^{k}$ y $C^{k}$ to encode the input data for each layer, with $k=1,\dots,K$ and $K$ the number of layers, that is

$$ u^{k+1}=o^{k}+u^{k}. $$

After going through all the layers, the final prediction will be

$$ \hat{a}=\text{Softmax}(Wu^{K+1})=\text{Softmax}(W(o^{K}+u^{K})).$$

Considering the same inmersion matrixes for each layer ($A^{1}=A^{2}=A^{3}=A and C^{1}=C^{2}=C^{3}=C$), and end-to-end memory network with 3 layers can be represented as in the following picture:

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2n22.png" width="700" height="300">
</p>

Doing this assumption, this model can be seen as a traditional recurrent neural network with two types of outputs: an internal output generated by the immersions $A^{i}$, which are the probabilities $p_{i}$ (weights), and an external output which is the response predicted from those probabilities and the immersions $C^{i}$.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2nasrnn.png" width="550" height="300">
</p>

### Key-Value Memory Networks (KV-MemNN)

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/kvmemnn2.png" width="700" height="580">
</p>
