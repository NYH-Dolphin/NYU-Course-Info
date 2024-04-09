# Lecture6 Computer Vision CNNs

## 1. Introduction to CNN

![image-20240407211240000](./Lecture6 Computer Vision CNNs.assets/image-20240407211240000.png)

### Biological inspiration

![image-20240407211346965](./Lecture6 Computer Vision CNNs.assets/image-20240407211346965.png)

The relationship between components of the visual system and the base operations of a convolutional neural network. Hubel and Wiesel discovered that simple cells (left, blue) have preferred locations in the image (dashed ovals) wherein they respond most strongly to bars of particular orientation. Complex cells (green) receive input from many simple cells and thus have more spatially invariant responses. These operations are replicated in a convolutional neural network (right). The first convolutional layer (blue) is produced by applying a convolution operation to the image. Specifically, the application of a small filter (gray box) to every location in the image creates a feature map.

### Translation invariance

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240407211416391.png" alt="image-20240407211416391" style="zoom:67%;" />

- Models are not “intelligent” beyond  what they’re made for
- Dense layers don’t model spatial relationships between pixels in an image: a change of position is like  “new data”

### The convolution operation

Images can be broken down into local patterns such as edges,  textures, etc.

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240407211441365.png" alt="image-20240407211441365" style="zoom:67%;" />

- Convolutional networks are built to account for  those patterns

## 2. Convolutional Neural Networks

![image-20240408155338273](./Lecture6 Computer Vision CNNs.assets/image-20240408155338273.png)

- Convolutional neural networks (CNNs), are a specific type of neural networks that are generally composed of convolutional, pooling and dense layers.

### The Convolution Operation

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408160234869.png" alt="image-20240408160234869" style="zoom:50%;" />

| <img src="./Lecture6 Computer Vision CNNs.assets/image-20240408155710847.png" alt="image-20240408155710847" style="zoom:80%;" /> | <img src="./Lecture6 Computer Vision CNNs.assets/image-20240408155728317.png" alt="image-20240408155728317" style="zoom:75%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240408155830257](./Lecture6 Computer Vision CNNs.assets/image-20240408155830257.png) | ![image-20240408155835888](./Lecture6 Computer Vision CNNs.assets/image-20240408155835888.png) |

| ![image-20240408160326474](./Lecture6 Computer Vision CNNs.assets/image-20240408160326474.png) | ![image-20240408160342441](./Lecture6 Computer Vision CNNs.assets/image-20240408160342441.png) | ![image-20240408160353214](./Lecture6 Computer Vision CNNs.assets/image-20240408160353214.png) | ![image-20240408160400057](./Lecture6 Computer Vision CNNs.assets/image-20240408160400057.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240408160436795](./Lecture6 Computer Vision CNNs.assets/image-20240408160436795.png) | ![image-20240408160442701](./Lecture6 Computer Vision CNNs.assets/image-20240408160442701.png) | ![image-20240408160449345](./Lecture6 Computer Vision CNNs.assets/image-20240408160449345.png) | ![image-20240408160456683](./Lecture6 Computer Vision CNNs.assets/image-20240408160456683.png) |
| ![image-20240408160503039](./Lecture6 Computer Vision CNNs.assets/image-20240408160503039.png) | ![image-20240408160509131](./Lecture6 Computer Vision CNNs.assets/image-20240408160509131.png) | ![image-20240408160517375](./Lecture6 Computer Vision CNNs.assets/image-20240408160517375.png) | ![image-20240408160523591](./Lecture6 Computer Vision CNNs.assets/image-20240408160523591.png) |
| ![image-20240408160529976](./Lecture6 Computer Vision CNNs.assets/image-20240408160529976.png) | ![image-20240408160534877](./Lecture6 Computer Vision CNNs.assets/image-20240408160534877.png) | ![image-20240408160542392](./Lecture6 Computer Vision CNNs.assets/image-20240408160542392.png) | ![image-20240408160551210](./Lecture6 Computer Vision CNNs.assets/image-20240408160551210.png) |

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408160638020.png" alt="image-20240408160638020" style="zoom:67%;" />

### What is convolutional layer about?

- One Convolutional layer == one **convolutional operation** (but **multichannel** usually; to be covered later)
- Convolutional operation is a 2-D shaped inner product.
    - It is about finding a **similar pattern**; as the output would be larger if two input vectors of inner product are similar.
- While finding a **similar pattern**, it also optimizes which pattern to find by **updating the weights** (=kernel, or filter) of the conv layer;

### Stride and Padding

#### Stride

A common stride is 2

| ![image-20240408160929737](./Lecture6 Computer Vision CNNs.assets/image-20240408160929737.png) | ![image-20240408160936616](./Lecture6 Computer Vision CNNs.assets/image-20240408160936616.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240408160944792](./Lecture6 Computer Vision CNNs.assets/image-20240408160944792.png) | ![image-20240408160953247](./Lecture6 Computer Vision CNNs.assets/image-20240408160953247.png) |

#### Padding

Using stride=2

| Valid                                                        | Same                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240408161334378](./Lecture6 Computer Vision CNNs.assets/image-20240408161334378.png) | ![image-20240408161357626](./Lecture6 Computer Vision CNNs.assets/image-20240408161357626.png) |
| ![image-20240408161406923](./Lecture6 Computer Vision CNNs.assets/image-20240408161406923.png) | ![image-20240408161422788](./Lecture6 Computer Vision CNNs.assets/image-20240408161422788.png) |
| ![image-20240408161428406](./Lecture6 Computer Vision CNNs.assets/image-20240408161428406.png) | ![image-20240408161433189](./Lecture6 Computer Vision CNNs.assets/image-20240408161433189.png) |
| ![image-20240408161440630](./Lecture6 Computer Vision CNNs.assets/image-20240408161440630.png) | ![image-20240408161444862](./Lecture6 Computer Vision CNNs.assets/image-20240408161444862.png) |

- Padding has an effect of how  we handle the borders

#### Choosing padding and stride

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408161544579.png" alt="image-20240408161544579" style="zoom:67%;" />

### Kernels and Channels

How does it work when it’s multichannel? e.g. input has 3 channels, output has 1 channel

#### Different kernels will learn different patterns

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408162014376.png" alt="image-20240408162014376" style="zoom:50%;" />

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408161804551.png" alt="image-20240408161804551" style="zoom:67%;" />

- This are the 16 filters of a convolutional layer.
- The kernel size is 5.

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408162023435.png" alt="image-20240408162023435" style="zoom:67%;" />

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408161946320.png" alt="image-20240408161946320" style="zoom:67%;" />

- You can look at the feature maps as they were images.
- This are the 16 feature maps corresponding to the convolution with  each filter.

#### Choosing filter size

- Actually a pretty fuzzy choice.
- There have been some papers focusing on filter sizes, but it’s **hard** to get a general lesson.
- For this class, let’s stick with (3, 3). (ref: this is the “vgg” way, and very common)

#### Choosing number of filters

- This is equal to the **max number of patterns we like a layer to learn**.
- Similar to the number of units in dense layer; or how “wide” layers should be.
- Again, no single rule exists.

### Max Pooling

- Similar to a convolution but without  learning the filter, we take the **max**.
- We only care about the max value;  we don’t care where it came from  exactly.
- This is why a ConvNet is “**invariant**”  to **small local changes**.

| ![image-20240408162402604](./Lecture6 Computer Vision CNNs.assets/image-20240408162402604.png) | ![image-20240408162423419](./Lecture6 Computer Vision CNNs.assets/image-20240408162423419.png) | ![image-20240408162428713](./Lecture6 Computer Vision CNNs.assets/image-20240408162428713.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240408162433561](./Lecture6 Computer Vision CNNs.assets/image-20240408162433561.png) | ![image-20240408162438251](./Lecture6 Computer Vision CNNs.assets/image-20240408162438251.png) | ![image-20240408162443691](./Lecture6 Computer Vision CNNs.assets/image-20240408162443691.png) |

- Needed to increase the Receptive Field of the network and decrease its size

### Last: Dense Layer

![image-20240408162856562](./Lecture6 Computer Vision CNNs.assets/image-20240408162856562.png)

- Idea: learn hierarchies of patterns (from edges to concepts) in an effective manner.

### Key characteristics of CNNs

#### They can learn spatial hierarchies of patterns

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408162954821.png" alt="image-20240408162954821" style="zoom:67%;" />

- CNNs’ layers “break down” the image in its main parts,  and learn to combine them in relevant ways in deeper layers

#### The patterns they learn are translation invariant (to some extent)

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408163050846.png" alt="image-20240408163050846" style="zoom:67%;" />

- CNNs will be able to  understand similar patterns  across the image.

### Application of CNN

#### LeNet (1989)

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408163118734.png" alt="image-20240408163118734" style="zoom: 50%;" />

#### AlexNet (2012)

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408163139623.png" alt="image-20240408163139623" style="zoom:67%;" />

- One of the earliest successful CNNs

#### VGGNet (2014)

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408163206961.png" alt="image-20240408163206961" style="zoom:67%;" />

#### ResNet (2016)

<img src="./Lecture6 Computer Vision CNNs.assets/image-20240408163552020.png" alt="image-20240408163552020" style="zoom:80%;" />

- Because it’s difficult to train networks that are too deep.. We added skip connections.

## 3. Wrap Up

### Convolution

- Translation invariant
- Learn hierarchies of patterns
- Main parameters
    - Number of filters
    - Kernel (or filer) size
    - Stride
    - Padding

### Max Pooling

- Needed to reduce network size and overfitting
- Needed to increase receptive field

### Remember this terms

- Convolutional Neural Network
- Filter
- Kernel
- Max-pooling
- Stride
- Padding
- Convolution
- Translation invariance
