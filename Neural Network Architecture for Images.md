#
# Neural Network Architectures for Images

![](RackMultipart20210222-4-1mdl20f_html_cb55ddb5edd60516.gif)

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
