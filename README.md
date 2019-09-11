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
and that its hidden layer is connected to itself, apart from to the output layer. Let <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/>, <img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/> and <img src="/tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/> be the number of 
existing neurons in the input, hidden and output layer, respectively. Consider <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> as the indicator of the observation times. Defining

<p align="center"><img src="/tex/2f14a6ffc287d5e36cb9c0d5fda26164.svg?invert_in_darkmode&sanitize=true" align=middle width=675.6170553pt height=308.28251685pt/></p>

A recurrent neural network can be defined by the following expression:

<p align="center"><img src="/tex/677d46fdacb690a248dc4e9ea5989d0d.svg?invert_in_darkmode&sanitize=true" align=middle width=495.3941651999999pt height=47.1348339pt/></p>


From now on, the following matrix notation will be used in order to simplify the formulas defining the networks. This representation can be seen as considering that there is only one node per each layer.

<p align="center"><img src="/tex/957908aa970145df6161eb5b5b59650e.svg?invert_in_darkmode&sanitize=true" align=middle width=675.61705035pt height=292.74160905pt/></p>


So, considering this notation, the recurrent neural network can be defined by the following expression:

<p align="center"><img src="/tex/9d19327353a52171d5f7e3e0c48578fc.svg?invert_in_darkmode&sanitize=true" align=middle width=381.4622526pt height=18.312383099999998pt/></p>


If several hidden layers exist, a layer will receive as input the initial inputs, in addition to the outputs from the previous layer,
and the final output layer will receive as inputs the outputs from all hidden layers. This makes the training of deep networks easier due to the number of processing steps from the bottom to the top of the network, and also it allows to soften the gradient vanishing problem.
A representation of a network with several hidden layers can be seen in the next picture:


<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/deeprnn.png" width="700" height="450">
</p>


### Long Short term Memory Networks (LSTM)

LSTM are recurrent neural networks whose neurons in the hidden layer are replaced by memory cells. These cells have a 'gates system' that decides what information should be stored in the memory, what information should be forgotten and what should be transferred to the rest of the layers. Typically, the activation functions considered in this kind of networks are the sigmoid function and the hyperbolic tangent. The structure of LSTM hidden layers can be represented, for a time instant <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>, as follows:

<p align="center"><img src="/tex/1cc4f9b51a6b29aed21708b9d2832290.svg?invert_in_darkmode&sanitize=true" align=middle width=345.39052845pt height=214.86757709999998pt/></p>

where <img src="/tex/627a293fc39b1bda314d1398ab3ea145.svg?invert_in_darkmode&sanitize=true" align=middle width=14.360779949999989pt height=14.15524440000002pt/> is the vector representing the entrance to the corresponding layer at instant <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>, <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/> represents the sigmoid function, <img src="/tex/18ae17f171d82e313dff61306c48eb47.svg?invert_in_darkmode&sanitize=true" align=middle width=7.2773926499999915pt height=14.15524440000002pt/> the hyperbolic tangent function, and <img src="/tex/8ff2aecdf4cb9f6eb0b3ce2ce568889f.svg?invert_in_darkmode&sanitize=true" align=middle width=22.07767979999999pt height=22.465723500000017pt/>, <img src="/tex/afbead5a696c02298a701b50adebc84c.svg?invert_in_darkmode&sanitize=true" align=middle width=22.07767979999999pt height=22.465723500000017pt/> are the weight matrixes that we need to estimate. In each instant of time, based on <img src="/tex/627a293fc39b1bda314d1398ab3ea145.svg?invert_in_darkmode&sanitize=true" align=middle width=14.360779949999989pt height=14.15524440000002pt/>, the LSTM generates the corresponding <img src="/tex/78e9eed698228f1de2177c97cc17493b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.436907649999991pt height=22.831056599999986pt/> y <img src="/tex/ea4b2135ddcd621fb20b2650325b4157.svg?invert_in_darkmode&sanitize=true" align=middle width=19.398893249999993pt height=14.15524440000002pt/>.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/LSTM2.png" width="450" height="320">
</p>

### Gated Recurrent Units (GRU)

Gated Recurrent Units, GRU, have less parameters than LSTM. They combine the input and forget gates into a unique one called update gate, and also the inner memory with the hidden state. Its structure is defined by the following expressions:

<p align="center"><img src="/tex/e7b689ef0f1fb856d6ddd9210806df2e.svg?invert_in_darkmode&sanitize=true" align=middle width=536.80176165pt height=95.49050444999999pt/></p>


where <img src="/tex/627a293fc39b1bda314d1398ab3ea145.svg?invert_in_darkmode&sanitize=true" align=middle width=14.360779949999989pt height=14.15524440000002pt/> is the vector representing the entrance to the corresponding layer at instant <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>, <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/> represents the sigmoid function, <img src="/tex/18ae17f171d82e313dff61306c48eb47.svg?invert_in_darkmode&sanitize=true" align=middle width=7.2773926499999915pt height=14.15524440000002pt/> the hyperbolic tangent function, and <img src="/tex/8ff2aecdf4cb9f6eb0b3ce2ce568889f.svg?invert_in_darkmode&sanitize=true" align=middle width=22.07767979999999pt height=22.465723500000017pt/>, <img src="/tex/afbead5a696c02298a701b50adebc84c.svg?invert_in_darkmode&sanitize=true" align=middle width=22.07767979999999pt height=22.465723500000017pt/> are the weight matrixes that we need to estimate. In each instant of time, based on <img src="/tex/627a293fc39b1bda314d1398ab3ea145.svg?invert_in_darkmode&sanitize=true" align=middle width=14.360779949999989pt height=14.15524440000002pt/>, the GRU generates the corresponding <img src="/tex/78e9eed698228f1de2177c97cc17493b.svg?invert_in_darkmode&sanitize=true" align=middle width=14.436907649999991pt height=22.831056599999986pt/>.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/GRU.png" width="450" height="350">
</p>

## Memory Networks

Memory networks, as the name implies, consist of a memory <img src="/tex/166e68cd93fe543dc48dfa0d8e308f1e.svg?invert_in_darkmode&sanitize=true" align=middle width=15.75334034999999pt height=14.611878600000017pt/> (an array of objects named as <img src="/tex/33aefc9781e07999d12cf46587fb46b2.svg?invert_in_darkmode&sanitize=true" align=middle width=20.40423989999999pt height=14.611878600000017pt/>) on which we can read and write information, and four components. The input component, <img src="/tex/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode&sanitize=true" align=middle width=8.515988249999989pt height=22.465723500000017pt/>, is the one in charge of encoding the input <img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> into the corresponding internal representation, <img src="/tex/bd84221165cd4b6441add4fe4ef28e0c.svg?invert_in_darkmode&sanitize=true" align=middle width=30.69638714999999pt height=24.65753399999998pt/>. The generalization component, <img src="/tex/5201385589993766eea584cd3aa6fa13.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>, updates every memory <img src="/tex/2a55b86ae85d759fd3e7b6bee1f72286.svg?invert_in_darkmode&sanitize=true" align=middle width=19.083998999999988pt height=14.15524440000002pt/> with the new input: <img src="/tex/8c9c5b88828ff14e2e8598e7204b6a0e.svg?invert_in_darkmode&sanitize=true" align=middle width=148.50099659999998pt height=24.65753399999998pt/>, <img src="/tex/bb01a0c98f80bbedd6021a97dc07c9b3.svg?invert_in_darkmode&sanitize=true" align=middle width=96.46192049999999pt height=22.831056599999986pt/>. The output component, <img src="/tex/9afe6a256a9817c76b579e6f5db9a578.svg?invert_in_darkmode&sanitize=true" align=middle width=12.99542474999999pt height=22.465723500000017pt/>, is in charge of processing an output <img src="/tex/7e1096128b080021db736ec4d7400387.svg?invert_in_darkmode&sanitize=true" align=middle width=7.968051299999991pt height=14.15524440000002pt/> from the new input <img src="/tex/bd84221165cd4b6441add4fe4ef28e0c.svg?invert_in_darkmode&sanitize=true" align=middle width=30.69638714999999pt height=24.65753399999998pt/> and the memory <img src="/tex/166e68cd93fe543dc48dfa0d8e308f1e.svg?invert_in_darkmode&sanitize=true" align=middle width=15.75334034999999pt height=14.611878600000017pt/>, <img src="/tex/fbd84315e487561bfa10a2006aa8396c.svg?invert_in_darkmode&sanitize=true" align=middle width=109.42215074999997pt height=24.65753399999998pt/>. Finally, the response component <img src="/tex/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.60847334999999pt height=22.465723500000017pt/> decodes the output <img src="/tex/7e1096128b080021db736ec4d7400387.svg?invert_in_darkmode&sanitize=true" align=middle width=7.968051299999991pt height=14.15524440000002pt/> providing an interpretable answer <img src="/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/>, like a text or an action: <img src="/tex/3a96a8989d48d13489cc9501b3ac1005.svg?invert_in_darkmode&sanitize=true" align=middle width=63.15254054999999pt height=24.65753399999998pt/>. In any of these components, any machine learning model could be used, but in this case we will focus on having a recurrent neural network in some of the components, thus obtaining a memory network.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2.png" width="550" height="220">
</p>

In the basic model, <img src="/tex/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode&sanitize=true" align=middle width=8.515988249999989pt height=22.465723500000017pt/> gets an input <img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> that can be encoded into an internal representation before moving forward to the component <img src="/tex/5201385589993766eea584cd3aa6fa13.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>, and this component will store it in the next available memory slot. The neural network is introduced in the component <img src="/tex/9afe6a256a9817c76b579e6f5db9a578.svg?invert_in_darkmode&sanitize=true" align=middle width=12.99542474999999pt height=22.465723500000017pt/>, that in the case of question answering, will be the one in charge of finding the <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> supporting sentences that answer the matter in question, being <img src="/tex/2ba1df65091b7034fd68f80dfd71ddfe.svg?invert_in_darkmode&sanitize=true" align=middle width=45.99298274999999pt height=22.831056599999986pt/> and <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> the number of memory slots. <img src="/tex/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.60847334999999pt height=22.465723500000017pt/> will generate a response from the input <img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/> (a question in this case) and the selected supporting memories. For <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> recurrences, the schema of a memory neural network can be represented as follows:


<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memnn21.png" width="550" height="320">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memnn22.png" width="600" height="230">
</p>


### End-to-End Memory Networks (MemN2N)

This network follows the same structure as the one already introduced, but it is trained end-to-end. This training strategy consists of training the whole system at the same time, from the input modules to the output ones. This implies that the network needs less supervision during the training phase, done by backpropagation, so it can be applied in more realistic scenarios. In this model, the recurrence can be found in the multiple accesses to the memory before generating an output.

We will explain how it works when applying it to find answers about a specific question in a text:

Consider <img src="/tex/073c6f4d0832ec9d6975a9d502dfa84d.svg?invert_in_darkmode&sanitize=true" align=middle width=70.81987439999999pt height=14.15524440000002pt/> the set of input data that need to be stored in the memory, <img src="/tex/d5c18a8ca1894fd3a7d25f242cbe8890.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928106449999989pt height=14.15524440000002pt/> the question and <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> its answer. The model consists of three phases:

<p align="center"><img src="/tex/bd588347dc1b8ebf2c9892891419da34.svg?invert_in_darkmode&sanitize=true" align=middle width=680.18321745pt height=306.06179505pt/></p>


The model learns the matrixes <img src="/tex/ed2738ef0c65ae7cd917d49e69a57ef5.svg?invert_in_darkmode&sanitize=true" align=middle width=77.35950584999999pt height=22.465723500000017pt/> by minimizing the standard cross entropy loss between <img src="/tex/dc96518a9dfd5d97cebc61d4e90ac25b.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=22.831056599999986pt/> and the real answer <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/>.




<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2n21.png" width="550" height="300">
</p>



#### Modifications
If we want the model to make multiple hops, it needs to be built with multiple layers instead of just one. This way the model will provide better results when predicting the answers, since it is able to access the memory multiple times. How it works is the same, but now we will have different matrixes <img src="/tex/10f455885705aed72da7fdac176a9292.svg?invert_in_darkmode&sanitize=true" align=middle width=19.594827449999993pt height=27.91243950000002pt/> y <img src="/tex/67d005227129ec89df883967779cf10f.svg?invert_in_darkmode&sanitize=true" align=middle width=20.19066719999999pt height=27.91243950000002pt/> to encode the input data for each layer, with <img src="/tex/d6f3b6e6a6da55dfc79b92d247f3359a.svg?invert_in_darkmode&sanitize=true" align=middle width=90.87862409999998pt height=22.831056599999986pt/> and <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> the number of layers, that is

<p align="center"><img src="/tex/e07eb93f49385c93c4d6dd01b0ebba51.svg?invert_in_darkmode&sanitize=true" align=middle width=114.27138525pt height=16.1480121pt/></p>

After going through all the layers, the final prediction will be

<p align="center"><img src="/tex/1454995a7d1678e6d026b72722c4687d.svg?invert_in_darkmode&sanitize=true" align=middle width=351.32975295pt height=18.7598829pt/></p>

Considering the same inmersion matrixes for each layer (<img src="/tex/cf13f5f01c24e507a246a8566e4fbc98.svg?invert_in_darkmode&sanitize=true" align=middle width=303.87828735pt height=26.76175259999998pt/>), and end-to-end memory network with 3 layers can be represented as in the following picture:

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2n22.png" width="700" height="280">
</p>

Doing this assumption, this model can be seen as a traditional recurrent neural network with two types of outputs: an internal output generated by the immersions <img src="/tex/b677d48b5606d5dc92be4c77a4fdbf85.svg?invert_in_darkmode&sanitize=true" align=middle width=16.97969789999999pt height=27.15900329999998pt/>, which are the probabilities <img src="/tex/8417860712b21d041c2fe6a6f3140369.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92146679999999pt height=14.15524440000002pt/> (weights), and an external output which is the response predicted from those probabilities and the immersions <img src="/tex/9dea2748c817a05dc77c9ab3a38fa722.svg?invert_in_darkmode&sanitize=true" align=middle width=17.57553764999999pt height=27.15900329999998pt/>.

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/memn2nasrnn.PNG" width="780" height="500">
</p>


### Key-Value Memory Networks (KV-MemNN)

<p align="center">
<img src="https://github.com/AndreaAI/Deep-Learning-Models-for-text-comprehension/blob/master/images/kvmemnn2.png" width="700" height="580">
</p>
