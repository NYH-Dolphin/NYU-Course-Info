# Lecture3 Engine Neural Network

## 1. How does learning happen?

### Kernel and Bias

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212144408497.png" alt="image-20240212144408497" style="zoom:67%;" />
$$
\text{output} = \text{relu}(\text{dot}(\text{input}, W) +b)
$$

- We saw that dense layers are these combination of operations, where $W$ and $b$ are **parameters** of the layer
- Those two matrices are called the **kernel** and **bias** attributes of  the dense layer
- Initially, they are **randomized**
    - i.e.  assigned random values

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212144449002.png" alt="image-20240212144449002" style="zoom:80%;" />

### Training loop

1. Draw a batch of training samples `x`, and  corresponding labels `y_pred`.

![image-20240212144547294](./Lecture3 The Engine Neural Network.assets/image-20240212144547294.png)

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212144604058.png" alt="image-20240212144604058" style="zoom:80%;" />

```python
batch = train_images[:128]
batch = train_images[128:256]
n = 3
batch = train_images[128*n:128*(n+1)]
```

2. Run the model on `x`  (**forward pass**) to obtain predictions `y_pred`

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212144707139.png" alt="image-20240212144707139" style="zoom:67%;" />

3. Compute the **loss** of the model on the batch, a measure of the mismatch between `y_pred` and `y_true`
4. Update all weights of the  model in a way that slightly  reduces the loss on this  batch.

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212144754432.png" alt="image-20240212144754432" style="zoom:67%;" />

## 2. How to Update Weight?

### Weight initialization: Randomly

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212145151257.png" alt="image-20240212145151257" style="zoom:67%;" />

- A linear layer with
    - 3 inputs
    - 2 outputs
    - its weight (w) is a (2 x 3) matrix (and a bias term (b))

- **This layer wouldn’t do anything useful**.
    - (I.e., a randomly initialized model for MNIST classification would show about 10% accuracy)

### Update the weights - A naive approach (X)

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212145246146.png" alt="image-20240212145246146" style="zoom:67%;" />

- Change one coefficient at a time,
    - e.g. increase its value by a little.
- Do the forward pass again, see if  the loss improved.

| Weight Adjustment                                 | Loss Computation                                             |
| ------------------------------------------------- | ------------------------------------------------------------ |
| Weight Initialization                             | <img src="./Lecture3 The Engine Neural Network.assets/image-20240212145312235.png" alt="image-20240212145312235" style="zoom:67%;" /> |
| Increase the first element in $W$, loss increases | <img src="./Lecture3 The Engine Neural Network.assets/image-20240212145332878.png" alt="image-20240212145332878" style="zoom:67%;" /> |
| Decrease the first element in $W$, loss decreases | <img src="./Lecture3 The Engine Neural Network.assets/image-20240212145354239.png" alt="image-20240212145354239" style="zoom:67%;" /> |

- This is terribly inefficient
- For a middle size network, e.g.  400k parameters, this is 800k forward passes
- In other words, we’d like a method that can **update all the weights** (slightly increase or decrease the value in a way that the model gets **better** (show a lower loss)

### Gradient Descent

> Invented in 1847 by French mathematician  Louis-Augustin Cauchy
>
> It is an optimization technique that powers modern neural networks

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212145655558.png" alt="image-20240212145655558" style="zoom:67%;" />

- Core idea: The “**gradient**” encodes information about what **small change to each coefficient** (i.e. positive or negative) will **decrease the loss the most**
- Using that information, you can change the coefficients iteratively until finding **a minimum value** for the loss

#### How

| ![image-20240212150025614](./Lecture3 The Engine Neural Network.assets/image-20240212150025614.png) |
| ------------------------------------------------------------ |
| ![image-20240212145751098](./Lecture3 The Engine Neural Network.assets/image-20240212145751098.png) |
| ![image-20240212145808794](./Lecture3 The Engine Neural Network.assets/image-20240212145808794.png) |
| ![image-20240212145818011](./Lecture3 The Engine Neural Network.assets/image-20240212145818011.png) |
| ![image-20240212145902782](./Lecture3 The Engine Neural Network.assets/image-20240212145902782.png) |

#### Naive Gradient Descent

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212150521393.png" alt="image-20240212150521393" style="zoom:67%;" />

Given a function $l(w)$ that you want to minimize with respect to $w$, do:

1. Initialize $w_{iter}=w_0$ in a random point
2. In a loop, do:
    - $w_{iter} = w_{iter} - lr \cdot rc$
    - where $rc = \frac{l(w_{iter}+\epsilon)-l(w_{iter})}{\epsilon}$
    - $lr$: **learning rate** (or **step size** in optimization)
3. Repeat step 2 until reach N iterations

## 3. Gradient

### What is Gradient?

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212150609824.png" alt="image-20240212150609824" style="zoom:80%;" />

### 1D → N-dim parameters

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212150739745.png" alt="image-20240212150739745" style="zoom:67%;" />

- The 2D plots so far
    - parameter space on x-axis (1-dim param)
    - loss function on y-axis
- Actual neural network models have **many parameters** (like, thousands or millions or billions)
    - parameter space is **N-dim** (i.e. w is a N-dim array)
    - loss function `y=f(input, w)`

### GD in Tensors

When working with tensors, GD is applied to each element of the tensor in the same way as before.

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212150832215.png" alt="image-20240212150832215" style="zoom:80%;" />

To know the value of $l(W_{iter})$ we need to do a forward pass, so actually $l(W_{iter}) = l(W_{iter},X)$ where is the input data.

### Stochastic Gradient Descent

We can’t fit all the data in memory at once to compute the gradient. We use **batches of data** instead

Stochastic Gradient Descent (SGD) is **GD over batches**. 

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212151016313.png" alt="image-20240212151016313" style="zoom:67%;" />

- where $l(X_{iter},W_{iter})$ is the loss over one batch of data $X_{iter}$

#### Variants and Extensions of SGD

Many variants and extensions of SGD incorporating many tricks

- SGD with momentum
- RMSprop
- Adam, AdamW
- Adagrad

### Learning Rate (aka step size)

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212151218676.png" alt="image-20240212151218676" style="zoom:67%;" />

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212151211355.png" alt="image-20240212151211355" style="zoom:67%;" />

- If **too big**, SGD might not converge (find the minimum)
- If **too small**, SGD will take a lot of iterations (and time)  to converge. If number of iterations is not enough, it won’t find the solution.
- Safe bet: **start with a biggish and a few iterations,  observe a batch of validation data and tune it**.

![image-20240212151255305](./Lecture3 The Engine Neural Network.assets/image-20240212151255305.png)

### Momentum

In DNNs there are many “valleys” where SGD can get stuck, and momentum helps it to “keep going”.

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212151342519.png" alt="image-20240212151342519" style="zoom:67%;" />

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212151326263.png" alt="image-20240212151326263" style="zoom:80%;" />

![image-20240212151409648](./Lecture3 The Engine Neural Network.assets/image-20240212151409648.png)

### Local Minimum

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212151512217.png" alt="image-20240212151512217" style="zoom:67%;" />

- SGD is not infallible
- Momentum typically helps in this cases, but  also not infallible
- Good news: local minimums in DNNs are not that bad

### Backpropagation

Backpropagation is a clever trick to compute the gradient of the loss with respect to the parameters through multiple layers (=deep learning models!)

<img src="./Lecture3 The Engine Neural Network.assets/image-20240212151547073.png" alt="image-20240212151547073" style="zoom:67%;" />

- **Main idea**: all the **gradients** of the tensor operations in the network **are known** in advance by TensorFlow, Keras, PyTorch.  Backpropagation combines the gradients of each layer to form the gradient of the loss.

### Takeaways

- SGD is very sensitive to the learning rate. Momentum often helps
- What is important when choosing an optimizer?
    - Choose an optimizer with momentum and some trick for adjusting the learning rate (e.g. Adam)
    - Choose a **big learning rate to start with** (e.g. 0.01 with Adam). 
        - If network doesn’t converge, make it smaller. 
        - Bigger $lr$ speeds things up. 
        - Always monitor the loss!.
    - Choose **a small number of epochs to start** (e.g. 10) and a small part of the data. 
        - If SGD (or variants)  converges, augment the data and/or epochs until you reach the size of your problem. 
        - If it doesn’t converge, you needs more epochs

## 4. Remember these terms

- Training loop
- Batches
- Optimizer
- Loss function
- Gradient
- Gradient Descent
- Stochastic Gradient  Descent (SGD)
- Learning rate
- Number of iterations
- Momentum
- Back propagation
- Convergence
- Global minimum
- Local minimum

### Training in a nutshell

![image-20240212151935626](./Lecture3 The Engine Neural Network.assets/image-20240212151935626.png)