# Lecture7 Transformer and Language Model

> Intro to Large Language Models by Andrej Karpathy https://www.youtube.com/watch?si=SSCVcFyY9FduwHbW&v=zjkBMFhNj_g&feature=youtu.be

## 1. Introduction

### Traditional Language models

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408164533122.png" alt="image-20240408164533122" style="zoom:80%;" />

LMs take text as input and does something useful such as

- classification
- named entity recognition
- translation
- completion
- summarization
- Q&A
- chat

### Large Language Models

Large language models (e.g., GPT, llama, Gemini) are so powerful that then can various  natural language tasks very well

![image-20240408164650870](./Lecture7 Transformer and Language Model.assets/image-20240408164650870.png)

### Old <-> New Language Models

![image-20240408164740668](./Lecture7 Transformer and Language Model.assets/image-20240408164740668.png)

- bag-of-word: collect a bunch of positive or negative words

Traditional Models are Small Models

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408165512949.png" alt="image-20240408165512949" style="zoom:67%;" />

- With relatively limited resources (HW and data), we could handle **smaller** models only
- Each model was trained for a **specific task** e.g. summarization, sentient analysis, etc

## 2. Language As Data

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408165626734.png" alt="image-20240408165626734" style="zoom:67%;" />

- **image**: (row, col, channel), pixel with colors
- **text**: a sequence of characters

### Neural Networks for Sequences

| Illustration                                                 | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./Lecture7 Transformer and Language Model.assets/image-20240408165928222.png" alt="image-20240408165928222" style="zoom:67%;" /> |                                                              |
| <img src="./Lecture7 Transformer and Language Model.assets/image-20240408170048843.png" alt="image-20240408170048843" style="zoom:67%;" /> | Split the sentence into sequence of words                    |
| <img src="./Lecture7 Transformer and Language Model.assets/image-20240408170211198.png" alt="image-20240408170211198" style="zoom:67%;" /> | Sequences with Single output<br />If sentient analysis: positive or negative? [0 or 1] |
| <img src="./Lecture7 Transformer and Language Model.assets/image-20240408170242140.png" alt="image-20240408170242140" style="zoom:67%;" /> | Sequences with Sequence output<br />If embedding learning: Each “Out n” is a vector that represents n-th word<br />Limit of this approach What if we want a chat bot? or translation? <br />I.e., The output is not 1:1 to input words but  about the whole input sequence. |
| <img src="./Lecture7 Transformer and Language Model.assets/image-20240408170407891.png" alt="image-20240408170407891" style="zoom: 50%;" /> | Sequence-to-sequence<br />Eg. Conversation                   |
| <img src="./Lecture7 Transformer and Language Model.assets/image-20240408170439034.png" alt="image-20240408170439034" style="zoom:50%;" /> | Sequence-to-sequence<br />Eg. Machine Translation            |

### Word to vector (embedding)

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408171113146.png" alt="image-20240408171113146" style="zoom:50%;" />

#### Embedding layers

Embedding layer is a (trainable) Dense layer that maps each word to a vector

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408170921940.png" alt="image-20240408170921940" style="zoom:50%;" />

- vocabulary size = 8 in this cases
- each vector using a one-hot? representation

#### Advanced: It’s not always “word”-based

There are **too many words** (like, millions of them) to handle. Words are not always completely independent to each other. e.g., “Speak” “Speaker”  “Speaking”, “Speakers”, ..

1. **Words**: used to be a choice: `speakers`
    - **Length**: A document can have few thousands words
    - **Vocab size**: even 50k is not enough; so out-of-vocab becomes an issue
2. **Characters**: `s`, `p`, `e`, `a`, `k`, `e`, `r`, `s`
    - **Length**: A document can have a hundred of thousands words (kinda too long)
    - **Vocab size**: English only has 26 alphabets: so it’s good, but kinda too small
3. **Sub-word tokens**: `speak`-`er`-`s` (split the word into smaller units that somehow meaningful)
    - Can be optimal in both length and vocab size

## 3. Sequence-specialized Models

What do we want from such a model?

- Can handle a long sequence
- Can **remember** the sequence of input vectors, **process** them, and perform tasks **based on** the inputs.

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408172019329.png" alt="image-20240408172019329" style="zoom:67%;" />

### RNNs: Recurrent Neural Networks

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408172053283.png" alt="image-20240408172053283" style="zoom: 50%;" />

#### Inside a recurrent unit

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408172134488.png" alt="image-20240408172134488" style="zoom:50%;" />

- Operations that are expected to mimic
    - save (input)
    - load (output)
    - delete (forget)
- .. as long as they were trained to do so, implicitly, because perhaps they would be helpful to do the task,
- i.e., lower the loss and perform the training task

#### Output of RNN

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408172302375.png" alt="image-20240408172302375" style="zoom:50%;" />

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408172316600.png" alt="image-20240408172316600" style="zoom:50%;" />

#### Pros and Cons of RNNs

- **Pros**
    - It can handle sequences with arbitrary length.. in theory
    - With some modifications (e.g., LSTMs), it simply performs really well
    - Memory-efficient as the input sequence gets longer
- **Cons**
    - Long sequence == A very **deep** network → Difficult to train
    - N-length sequence → N-times matrix multiplication → Large latency
    - The final internal state is supposed to remember everything in the past → Is it even possible

### Transformers

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408172548383.png" alt="image-20240408172548383" style="zoom:67%;" />

| First Paper about Transformers                               | GPT-3                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./Lecture7 Transformer and Language Model.assets/image-20240408172622322.png" alt="image-20240408172622322" style="zoom:67%;" /> | <img src="./Lecture7 Transformer and Language Model.assets/image-20240408172630687.png" alt="image-20240408172630687" style="zoom: 67%;" /> |

#### Everything connects to everything

| Illustration                                                 | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./Lecture7 Transformer and Language Model.assets/image-20240408172853751.png" alt="image-20240408172853751" style="zoom:50%;" /> | Input word vector, forming a 2d dimensional channel input    |
| <img src="./Lecture7 Transformer and Language Model.assets/image-20240408173205330.png" alt="image-20240408173205330" style="zoom:50%;" /> | Mutual “Relatedness” is  computed by computing &  comparing all the 4 x 4 = 16pairs,  when input length is 4. |

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408173235992.png" alt="image-20240408173235992" style="zoom: 50%;" />

#### Pros and Cons for Transformers

- **Pros**
    - It outperforms RNNs and easy to train
    - It take all the mutual relationship between words
- **Cons**
    - $n$ words → $n^2$ relationships to compute and store → expensive!
    - It doesn’t work with arbitrary length

#### To Learn More

Search for these

- Illustrated Transformers
- Annotated Transformers

## 4. Large Language Model

### LLM = A lot of transformer layers

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408173457430.png" alt="image-20240408173457430" style="zoom: 50%;" />

### Coming back to Tasks

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408173603828.png" alt="image-20240408173603828" style="zoom: 50%;" />

- Not ideal that we have to train $N$ models for $N$ tasks.

### Autoregressive training for Pre-training

> Autoregressive models are a type of statistical model used in time series forecasting and **sequence prediction**. 
>
> The core idea is that the **future state of a sequence depends solely on its previous states**. This is particularly powerful in fields like language modeling, where the sequence of words is predicted based on the preceding context.

As known as “next-token prediction”

<img src="./Lecture7 Transformer and Language Model.assets/image-20240408174238905.png" alt="image-20240408174238905" style="zoom:67%;" />

- At one step, the model  performs and learns from 6 token predictions. → Efficient!
- OpenAI and others trained LLMs with this objective;  using A LOT of data. We call it **LLM pre-training**.
- After pre-training, the model  becomes excellent at text  completion; called  “**foundational model**”

#### Next token prediction as pretraining task

- Used in all the modern LLMs (GPT, GPT2, GPT3, LLaMA, etc)
- Universally useful for some specific task.
- Effectiveness, relevance: That’s how we speak and how we think to speak
- Efficiency: If 2048 token length, we make 2048 predictions, get 2048 losses  (information about how the model is doing), make update related to 2048 of them!

#### Autoregressive training for finetuning

- Once the model is good at  text completion, we can  further train the model to **steer the direction of the answer**.
- A specific application is “instruction finetuning” i.e.,  **tune** the model to perform a given task.
- This makes GPT → ChatGPT.

### Summary

- Language Models have advanced a lot, and these days they’re really strong. 
- The modern language models consists of deep learning models that can handle sequential data. 
- Data processing is still a part of the work, although it’s becoming less and less important. 
- Language Models are not perfect!

#### What do they learn?

- Originally: to predict the next tokens
- Now: Perfect grammar and writing + Questionable logical thinking

#### Why is it so amazing?

- Text data
    -  A LOT on internet
    - Very cheap to get and handle (unlike images, audio, music)
    - Language is a crucial and complex representation of our knowledge; arguably  the most important invention of human species