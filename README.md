## Networks
### Description
This project is an attempt to implement pre-established NN architectures. Contents include **LeNet-5** and **Yolov1** (more to come).

### LeNet-5
I decided to first implement LeNet as means of understanding the origins of CNNs (sourced from [this paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)). Its architecture contains the following layers: <br />

|INDEX|TYPE|SHAPE|KERNEL SIZE|STRIDE|PADDING|ACTIVATION|
|-|:-:|:-:|:-:|:-:|:-:|:-:|
|0|INPUT|28x28|-|-|-|-|
|1|CONV|28x28x6|5|1|SAME|TANH|
|2|AVGPOOL|14x14x6|2|2|VALID|-|
|3|CONV|10x10x16|5|1|SAME|TANH|
|4|AVGPOOL|5x5x16|2|2|VALID|-|
|5|FC|120|-|-|-|TANH|
|6|FC|84|-|-|-|TANH|
|7|OUTPUT|10|-|-|-|SOFTMAX|

The default training parameters for the network are as follows:
- Number of epochs: **5**
- Batch size: **64**
- Optimizer: **Adam**
- Loss function: **Categorical Cross Entropy**
- Minimal learning rate: **0.00001**
- Learning rate reduction factor on plateau: **0.5** <br />

The implementation was achieved both in Keras and PyTorch. At the end of training, statistics will be graphed and displayed, followed by samples from the testing set along with their predicted labels and levels of confidence. The network trains on the Fashion MNIST dataset, learning to classify pieces of clothing: <br />

<p align="center">
  <img src="https://github.com/worthy11/networks/blob/main/img/fashion_mnist.png" alt="Sample from Fashion MNIST dataset"/>
</p>

### Yolov1
In order to explore one-stage detector models, I'm implementing the first version of Yolo (sourced from [this paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)). Its architecture can be found in `src/Yolov1/config.py` and is visualized as follows: <br />

<p align="center">
  <img src="https://github.com/worthy11/networks/blob/main/img/yolo_arc.png" alt="Sample from Fashion MNIST dataset"/>
</p>

The implementation is not yet complete - currently working on the loss function.