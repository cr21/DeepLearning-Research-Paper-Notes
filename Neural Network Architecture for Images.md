#
# Neural Network Architectures for Images

![](RackMultipart20210222-4-157n9zq_html_cb55ddb5edd60516.gif)

1. **Multi-Layer Perceptron (Fully Connected Networks**

- MLP can represent and fit almost any function (Universal approximation Theorome) with enough hidden layers Fully connected network can learn any function.
- We can think of MLP qualitatively doing some Non-Linear representation of the data to classification or another task.

**Problem with MLP**

- After the network is trained we can&#39;t alter the input size, we need to retrain the network again.
- MLP is not considering any geometric or location-specific information.
- MLP is permutation invariance, we can run random permutation of MNIST data to classify and MLP will find it hard to train it.
- This is not useful or very hard to train where tasks involved some location or geometric specific learning. Human and Animals using geometric and location information for object detection and classification task. MLP is not built to deal with this directly.

1. **Convolution Neural Network (CNN)**

- Local Parts of Images is having lots of information for image classification purposes.
- Convolution Neural network is built to take leverage of geometry and location information of images
- In Most CNN network we have series of Convolution layers follow by series of nonlinear layers and in some case follow series of Fully connected layers
- CNN architecture is well suited for Images, CNN is taking into account the locality and geometry of images. CNN architecture is translation invariance.
- CNN networks without any fully connected networks can handle arbitrary image sizes.

- **Vanilla CNN**
  - Vanilla CNN is a variant of CNN in which we only have a convolution layer, this type of network is useful for image restoration tasks.
  - This network can work on any Input size.
  - **Consideration :**
    - Be careful about receptive fields and the scale of images.
    - One pixel in the output image depends on the filter size of the previous layer, and the combination of large spatial positions in the previous layer till the input image. There are a set of input fields that ignite the pixel in the output image.
      - For Example, If we trained the network on lower resolution images and testing on higher resolution images then the receptive field will be low because of training on low-resolution images which will not make meaningful modifications to the final output.
      - Another example is the Dirt removal network, When Network trained on dirt removal task tested on the simple image, image with dirt, image with snow, and image with scratches, it is able to remove dirt successfully, but it did not remove snow and scratches even if this task is related to dirt removal.
      - **Blind Convolution was** run on images with Gaussian noise, noisy blur, motion blur, and other combinations of blur using 15 layers of convolution layers.

1. **Residual networks and Skip Connection**

- In Residual block contains many convolution layers, it learned the residual between its input and output.
- The residual network _ **allows the increasing depth** _ of the neural network.
- **Increasing depth of networks has the following benefits.**
  - It allows _ **large receptive fields** _
  - It can allow using of _ **smaller filter sizes** _ with many _ **deep layers** _ to get the comparable performance of shallow layers having a large filter.
  - Saving in parameter count
- **RESNET for super-resolution of Images or noise removal of images**.
  - **RESNET** is good at the successive refinement of noise removal and image resolution.
  - _ **Learning residual is sometimes easier than learning the full mapping** _ between low resolution and high-resolution images.

1. **Encoder-Decoder Network**

- This type of network has series of layers (mostly convolution layer that downsamples the data.
- Then this type of network has series of bottlenecks having low parameters and a low degree of freedom.
- Decode takes the input from these bottlenecks and upsamples the data and generates the other image or output data.
- **Encoder** : Convert the image into a set of latent variables or codes.
- **Decoder** : convert set of latent variables to output code or images.
- **BottleNeck:** forces the network to gain the semantic understanding of images. This bottleneck sensor has a very low degree of freedom, but they have to express the relation between input and output highest with a little degree of freedom.
- This type of network is useful in tasks where the semantic meaning of images or part of images or input data are very important than the row features or pixel.
- _ **The compressive nature of encoding and decoder may result in loss of details in the output of the decoder.** _

1. **U-Net ( Encoder-Decoder with Skip connection)**

- To overcome the problem of loss of details in the encoder and decoder network, _ **U-net comes up with Encoder and Decoder with Skip connections,** _ **Skip Connections** _ **preserve the details.** _
- _ **U-Net is useful for Image segmentation tasks** _ where the semantic understanding of images is required.

1. **Auto Encoder-Decoder**

- Auto Encoder and Decoder are useful for unsupervised representation learning.
- Collection of images are passed to network and representation of those images are learned.
- For Example, Compressive Auto Encoder will learn the representation of images that can be useful for image compression tasks.
- The compressive Autoencoder has beaten JPEG and JPEG2000.
- Auto Encoder-Decoder can be useful for Image super-resolution task.
- The network is trained on both low and high-resolution images and latent variables for both high and low-resolution images have been learned. Then this type of network uses those low and high-resolution latent encoded values to generate super-resolution images.
- _ **Representation of images learned in this unsupervised task can be useful for solving other supervised or unsupervised tasks in the future.** _

1. **Spare networks from learning you already know**

- Fully connected networks might not be a good approach for every task in our hands.
- For Example, Fully connected networks are not good in translation, rotation invariance. It is not like that Fully connected network can&#39;t learn this, but for Translation and rotation invariance learning we need to feed augmented data using scaled and rotated images to the network.
- Rather than using this approach, we can use a convolution neural network that is built to handle translation and scale-invariant.
- For example in the compress sensing task rather than directly learn from raw images, raw images can be transformed to some approximation of images, and then we can train that representation (rough approximation) to learn the final image.

**References:**

- This lecture is from Northeastern University&#39;s CS 7150 Summer 2020 class on Deep Learning, taught by Paul Hand.
- https://www.youtube.com/watch?v=ty6IKzM\_VaM&amp;ab\_channel=PaulHand

- **LeCun et al. 1998:**
  - LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick Haffner. &quot;Gradient-based learning applied to document recognition.&quot; Proceedings of the IEEE 86, no. 11 (1998): 2278-2324.

- **Krizhevsky et al. 2012:**
  - Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. &quot;Imagenet classification with deep convolutional neural networks.&quot; In Advances in neural information processing systems, pp. 1097-1105. 2012.
- **Simonyan and Zisserman 2014:**
  - Simonyan, Karen, and Andrew Zisserman. &quot;Very deep convolutional networks for large-scale image recognition.&quot; arXiv preprint arXiv:1409.1556 (2014).
- **Lucas et al. 2018:**
  - Lucas, Alice, Michael Iliadis, Rafael Molina, and Aggelos K. Katsaggelos. &quot;Using deep neural networks for inverse problems in imaging: beyond analytical methods.&quot; IEEE Signal Processing Magazine 35, no. 1 (2018): 20-36.
- **Eigen et al. 2013:**
  - Eigen, David, Dilip Krishnan, and Rob Fergus. &quot;Restoring an image taken through a window covered with dirt or rain.&quot; In Proceedings of the IEEE international conference on computer vision, pp. 633-640. 2013.
- **Harris et al. 2015:**
  - Hradiš, Michal, Jan Kotera, Pavel Zemcık, and Filip Šroubek. &quot;Convolutional neural networks for direct text deblurring.&quot; In Proceedings of BMVC, vol. 10, p. 2. 2015.
- **He et al. 2015:**
  - He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. &quot;Deep residual learning for image recognition.&quot; In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.
- **Ledig et al. 2017:**
  - Ledig, Christian, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken et al. &quot;Photo-realistic single image super-resolution using a generative adversarial network.&quot; In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4681-4690. 2017.
- **Kim et al. 2016:**
  - Kim, Jiwon, Jung Kwon Lee, and Kyoung Mu Lee. &quot;Accurate image super-resolution using very deep convolutional networks.&quot; In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1646-1654. 2016.
- **Patha et al. 2016:**
  - Pathak, Deepak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. &quot;Context encoders: Feature learning by inpainting.&quot; In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2536-2544. 2016.
- **Ronneberger et al. 2015:**
  - Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. &quot;U-net: Convolutional networks for biomedical image segmentation.&quot; In International Conference on Medical image computing and computer-assisted intervention, pp. 234-241. Springer, Cham, 2015
- **Theis et al. 2017:**
  - Theis, Lucas, Wenzhe Shi, Andrew Cunningham, and Ferenc Huszár. &quot;Lossy image compression with compressive autoencoders.&quot; arXiv preprint arXiv:1703.00395 (2017).

- **Zeng et al. 2015**
  - Zeng, Kun, Jun Yu, Ruxin Wang, Cuihua Li, and Dacheng Tao. &quot;Coupled deep autoencoder for single image super-resolution.&quot; IEEE transactions on cybernetics 47, no. 1 (2015): 27-37.

- **Mousavi and Baraniuk 2017:**

  - Mousavi, Ali, and Richard G. Baraniuk. &quot;Learning to invert: Signal recovery via deep convolutional networks.&quot; In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 2272-2276. IEEE, 2017.

- **Zbontar et al. 2018:**
  - Zbontar, Jure, Florian Knoll, Anuroop Sriram, Matthew J. Muckley, Mary Bruno, Aaron Defazio, Marc Parente et al. &quot;fastMRI: An open dataset and benchmarks for accelerated MRI.&quot; arXiv preprint arXiv:1811.08839 (2018).
