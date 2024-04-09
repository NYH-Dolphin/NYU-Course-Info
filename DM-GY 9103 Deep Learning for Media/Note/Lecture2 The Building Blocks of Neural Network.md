# Lecture2 The Building Blocks of Neural Network

## 1. How Does Learning Happen?

### Weights

<img src="./Lecture2 The Building Blocks of Neural Network.assets/image-20240212134024152.png" alt="image-20240212134024152" style="zoom: 50%;" />

- A layer is parametrized by its weights
- Weights = Some numbers;  A (lossy) memory of input data the model has seen so far
- (Machine) Learning: Find the right values for the weights of each layer such that the network solves the task.

### Loss function (~= Objective Function)

<img src="./Lecture2 The Building Blocks of Neural Network.assets/image-20240212134144362.png" alt="image-20240212134144362" style="zoom:67%;" />

- How do we know if the values of  the weights are good?
    - i.e., the network outputting the right answer?
- **Loss function**: compares our actual output vs. expected output
- **Loss score**: indicates how far the network is from a good guess
- `loss_score = loss_function(our out, desired out)`

### Optimizer to update weights

<img src="./Lecture2 The Building Blocks of Neural Network.assets/image-20240212134233336.png" alt="image-20240212134233336" style="zoom: 50%;" />

- How do we correct the values based on what we observe?
    - We use an optimizer to modify the weights towards minimizing the loss
    - i.e. making  the network guess closer to the target

## 2. Rest of the Content

Check the  [Lecture 2 Building_Blocks.ipynb](Lecture 2 Building_Blocks.ipynb) or [Colab](https://colab.research.google.com/drive/13ByhmWc__O2FoYi3qLfbV3UESlBfcyLa#scrollTo=KKa3sQZH3cqb) for the rest of the content

## 3. Remember these terms

- Dense layer
- Tensor
- Tensor operations
- Tensor addition
- Tensor dot product
- Tensor slice
- Rank
- Dimensions
- Optimizer
- Loss
- Compile a model
- Fit data to a model
- Accuracy
