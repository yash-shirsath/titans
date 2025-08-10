      

{0}------------------------------------------------

# **Titans: Learning to Memorize at Test Time**

Ali Behrouz<sup> $\dagger$ </sup>, Peilin Zhong<sup> $\dagger$ </sup>, and Vahab Mirrokni<sup> $\dagger$ </sup>

<sup>†</sup>Google Research

{alibehrouz, peilinz, mirrokni}@google.com

### Abstract

Over more than a decade there has been an extensive research effort of how effectively utilize recurrent models and attentions. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new neural long-term memory module that learns to memorize historical context and helps an attention to attend to the current context while utilizing long past information. We show that this neural memory has the advantage of a fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics, and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models. They further can *effectively* scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks compared to baselines.

### **Introduction** $\mathbf{1}$

"The true art of memory is the art of attention!"

Samuel Johnson, 1787

🔊 ransformers, pure attention-based architectures (Vaswani et al. 2017), have been firmly established as state-ofthe-art models in sequence modeling, mainly due to their in-context learning and ability to learn at scale (Kaplan et al. 2020). The primary building blocks of Transformers-attention modules-function as associative memory blocks (Bietti et al. 2024), where they learn to store key-value associations and retrieve them by computing pairwise similarity between queries (i.e., search signals) and keys (i.e., contexts). Accordingly, by design, the output of a Transformer is exclusively conditioned on the direct dependencies of tokens in the *current* context window. This accurate modeling of dependencies, however, comes with quadratic time and memory complexity in terms of the context length. In complex real-world tasks (e.g., language modeling (N. F. Liu et al. 2024), video understanding (C.-Y. Wu et al. 2019), long-term time series forecasting (H. Zhou et al. 2021)), the context window can become extremely large, making the applicability of Transformers challenging in these downstream tasks.

To overcome the scalability issue of Transformers, recent studies aim to design different variants of linear Transformers (Kacham, Mirrokni, and P. Zhong 2024; Katharopoulos et al. 2020; S. Yang, B. Wang, Shen, et al. 2024), where softmax is replaced by a kernel function in the attention (see  $\S2.1$  for details), resulting in a significant drop in memory consumption. Despite efficiency and the ability to scale to longer context, linear Transformers do not show competitive performance compared to Transformers as the kernel trick makes the model a linear recurrent network, in which the data is compressed into a matrix-valued states (Katharopoulos et al. 2020). This, however, brings a contradictory fact about linear recurrent (or linear Transformers) models: On one hand, we use these linear models to enhance scalability and efficiency (linear vs. quadratic complexity), whose advantages is appeared for very long context; On the other hand, a very long context cannot be properly compressed in a small vector-valued or matrix-valued states (S. Wang 2024).

{1}------------------------------------------------

Furthermore, beyond efficiency, most existing architectures-ranging from Hopfield Networks (Hopfield 1982) to LSTMs (Jürgen Schmidhuber and Hochreiter 1997) and Transformers (Vaswani et al. 2017)-face challenges when dealing with generalization, length extrapolation, and/or reasoning (Anil et al. 2022; Qin, Y. Zhong, and Deng 2024), all of which are inseparable parts of many hard real-world tasks. Although these architectures draw inspiration from the human brain, each of which are missing: (1) a crucial component for learning process—such as short-term memory, long-term memory, meta-memory, attending to current context, etc. (Cowan 2008); (2) how these components are interconnected systems that can operate independently; and/or (3) the ability to *actively* learn from data and memorize the abstraction of past history. We argue that in an effective learning paradigm, similar to human brain, there are *distinct* yet interconnected modules, each of which is responsible for a component crucial to the learning process.

### **Memory Perspective**

Memory is a fundamental mental process and is an inseparable component of human learning (Terry 2017). Without a properly functioning memory system, humans and animals would be restricted to basic reflexes and stereotyped behaviors. Accordingly, memory has been the inspiration for many seminal research in machine learning literature; e.g., Hopfield Networks (Hopfield 1982), LSTMs (Jürgen Schmidhuber and Hochreiter 1997), and Transformers (Vaswani et al. 2017).

Taking inspiration from the common definitions of memory and learning in neuropsychology literature (Okano, Hirano, and Balaban 2000), most existing architectures consider memory as a neural update caused by an input, and define learning as a process for acquiring effective and useful memory, given an objective. In this perspective, Recurrent Neural Networks (RNNs) (Williams and Zipser 1989) can be defined as models with a vector-valued memory module  $\mathcal{M}$  (also called hidden state) with two main steps: Given a new input  $x_t$  at time t, the model (1) updates the memory using a function  $f(\mathcal{M}_{t-1}, x_t)$ (with compression); and (2) retrieves the corresponding memory of input using a function  $q(\mathcal{M}_t, x_t)$  (see §2.1 for details). Similarly, Transformers can be seen as architectures with a growing memory and two similar steps. That is, the pair of key and value matrices acts as the model's memory, and the model: (1) updates the memory by appending the key and value to the memory (without compression), and (2) retrieves query vectors' corresponding memory by finding the similarity of query and key vectors, which is then used to weight the value vectors for the output.

This perspective, can help us better understand existing paradigms, their critical differences, and design more effective architectures. For example, the main difference between Transformers (Vaswani et al. 2017) and *linear* Transformers (Katharopoulos et al. 2020) is the memory structure as well as the memory updating step, in which linear Transformers compress the historical data into a fixed-size matrix-valued memory while Transformers keep all historical data (within the context length) without any compression. While both linear Transformers and linear RNNs (including state space models) compress the information in memory update step, the critical difference lies in the structure of the memory, where linear RNNs (vs. linear Transformers) use a vector-valued memory (vs. matrix-valued memory). Therefore, this perspective motivates us to ask: (Q1) What constitute a good structure for the memory? (Q2) What is a proper memory update mechanism? and (Q3) What is a good memory retrieval process?

Revisiting our understanding of human memory, it is neither a unitary process nor it serves a single function (Cowan 2008). In fact, memory is a confederation of systems–e.g., short-term, working, and long-term memory–each serving a different function with different neural structures, and each capable of operating independently (Willingham 1997). This fact motivates us to ask: (Q4) How to design an efficient architecture that incorporates different interconnected memory modules. Finally, storing a memory is a neural process that requires to encode and store the abstraction of the past. It can be over-simplification to assume a single vector or a matrix, whose parameters are encoding the data in a linear manner, are enough for storing long-term history. (Q5) Is a deep memory module needed to effectively store/remember long past?

## **Contributions and Roadmap**

In this paper, we aim to answer the above five questions by designing a long-term neural memory module, that can efficiently and effectively learn to memorize at test time. Building upon its design, we discuss how it can be incorporated into an architecture.

**Neural Memory** ( $\S$ 3). We present a (deep) neural long-term memory that (as a meta in-context model) learns how to memorize/store the data into its parameters at test time. Inspired by human long-term memory system (Mandler 2014), 

{2}------------------------------------------------

we design this memory module so an event that violates the expectations (being surprising) is more memorable. To this end, we measure the surprise of an input with the gradient of the neural network with respect to the input in *associative memory loss* (see §3.1 for details). To better handle the limited memory, we present a decaying mechanism that consider the proportion of memory size and the amount of data surprise, resulting in better memory management. We show that this decay mechanism is in fact the generalization of forgetting mechanism in modern recurrent models (Dao and Gu 2024; Gu and Dao 2024; S. Yang, Kautz, and Hatamizadeh 2024). Interestingly, we find that this mechanism is equivalent to optimizing a meta neural network with mini-batch gradient descent, momentum, and weight decay. Building upon tensorizing mini-batch gradient descent to use more matmul operations (Yu Sun et al. 2024), we present a fast and parallelizable algorithm to train our deep neural long-term memory.

**Titans Architectures** ( $\S4$ ). After designing the long-term neural memory, an important remaining question is how to effectively and efficiently incorporate memory into a deep learning architecture. We present Titans, a family of deep models that consists of three hyper-heads:  $(1)$  Core: this module consists of the short-term memory, and is responsible for the main flow of processing the data (we use attention with limited window size); (2) Long-term Memory: this branch is our neural long-term memory module that is responsible to store/remember long past; (3) Persistent Memory: this is a set of learnable but date-independent parameters that encodes the knowledge about a task. Finally, as a proof of concept, we present three variants of Titans, in which we incorporate memory as: (i) a context, (ii) a layer, and (iii) a gated branch.

**Experimental Results** (§5). We perform experimental evaluations on language modeling, commonsense reasoning, recallintensive, needle in haystack, time series forecasting, and DNA modeling tasks. We observe that our Titan architecture outperforms all modern recurrent models as well as their hybrid variants (combining with sliding-window attention) across a comprehensive set of benchmarks. Furthermore, Titans outperforms Transformers with the same context window, and show competitive performance with Transformers that use the entire context. This results are achieved while, contrary to Transformers, Titans scale to larger than 2M context window size.

### $\mathbf{2}$ **Preliminaries**

**Solution** in this section, we discuss the notation and some background concepts that we use though the paper. We let  $x \in \mathbb{R}^{N \times d_{in}}$  be the input,  $M$  be a neural network (neural memory module),  $Q$ ,  $K$ ,  $V$  be the the  $i$ -th segment. Through the paper, we abuse the notation and use subscripts to refer to a specific element of a matrix, vector, or segments. For example, we let  $S_j^{(i)}$  be the *j*-th token in the *i*-th segment. The only exception is subscripts with *t*, which we reserved to index recurrence over time, or the state of a neural network at a data sample x, we use  $\mathcal{N}(x)$  (resp.  $\mathcal{N}^*(x)$ ) to refer to the forward pass with (resp. without) weight adjustment. Also, we abuse the notation and use  $\mathcal{N}^{(k)}$  to refer to the *k*-th layer of the neural network. In the following, we first, discuss the backgrounds for attention and its efficient variants followed by a review of modern linear RNNs. Finally, we discuss a memory perspective of these architectures that motivates us to design Titans.

#### $2.1$ **Backgrounds**

Attention. Transformers (Vaswani et al. 2017) as the de facto backbone for many deep learning models are based on attention mechanism. Given input  $x \in \mathbb{R}^{N \times d_{in}}$ , causal attention computes output  $y \in \mathbb{R}^{N \times d_{in}}$  based on softmax over input dependent key, value, and query matrices:

$$
Q = xW_Q, \t K = xW_K, \t V = xW_V,
$$
 (1)

$$
\mathbf{y}_i = \sum_{j=1}^i \frac{\exp\left(\mathbf{Q}_i^{\top} \mathbf{K}_j / \sqrt{d_{\text{in}}}\right) \mathbf{V}_j}{\sum_{\ell=1}^i \exp\left(\mathbf{Q}_i^{\top} \mathbf{K}_\ell / \sqrt{d_{\text{in}}}\right)},
$$
(2)

where  $W_O$ ,  $W_K$ , and  $W_V \in \mathbb{R}^{d_{in} \times d_{in}}$  are learnable parameters. Despite the power and effectiveness in recall, transformers need at least  $N \times d$  operators to calculate the output, resulting in larger memory consumption and lower-throughput for longer sequences.

**Efficient Attentions.** To improve the memory consumption and throughput of softmax attention for longer sequences, various studies focused on I/O aware implementations of attention (Dao 2024; Dao, D. Fu, et al. 2022), designing more

{3}------------------------------------------------

efficient attention mechanisms by sparsifying the attention matrix (B. Chen et al. 2021; Choromanski et al. 2021; Dai et al. 2019), approximating the softmax (Arora et al. 2024), or developing kernel-based (linear) attentions (Aksenov et al. 2024; Kacham, Mirrokni, and P. Zhong 2024; Schlag, Irie, and Jürgen Schmidhuber 2021; S. Yang, B. Wang, Shen, et al. 2024). In this part, we focus on the later, i.e., linear attentions, where the softmax in standard attention is replaced with an alternative kernel function  $\phi(., .)$ , such that  $\phi(x, y) = \phi(x)\phi(y)$ . Accordingly, the attention can be written as:

$$
y_i = \sum_{j=1}^i \frac{\phi(Q_i^\top K_j)}{\sum_{\ell=1}^i \phi(Q_i^\top K_\ell)} V_j = \sum_{j=1}^i \frac{\phi(Q_i)^\top \phi(K_j)}{\sum_{\ell=1}^i \phi(Q_i)^\top \phi(K_\ell)} V_j = \frac{\phi(Q_i)^\top \sum_{j=1}^i \phi(K_j) V_j}{\phi(Q_i)^\top \sum_{\ell=1}^i \phi(K_\ell)},
$$
(3)

resulting in a higher-throughput as terms  $\sum_{j=1}^{i} \phi(K_j)$  and  $\sum_{\ell=1}^{i} \phi(K_\ell)$  are re-using in each step. When choosing the kernel as identity matrix (Yutao Sun et al. 2023), the above formulation can also be written in a recurrent format:

$$
\mathcal{M}_t = \mathcal{M}_{t-1} + K_t^{\top} V_t, \tag{4}
$$

$$
\mathbf{y}_t = Q_t \mathcal{M}_t \,, \tag{5}
$$

which allows efficient inference for linear attentions.

Modern Linear Models and Their Memory Perspective. As discussed earlier, one can define learning as a process for acquiring effective and useful memory. Building upon this, one can see the hidden state of Recurrent Neural Networks (RNNs) as a memory unit, which the model aims to compress the information into. Accordingly, in a general form of recurrent neural network, the hidden state can be treated as a memory unit and the recurrence process can be split into the *read* and *write* operations in the memory unit. That is, we let  $x \in \mathbb{R}^{N \times d_{in}}$  be the input,  $M \in \mathbb{R}^d$  is the memory unit, and  $y \in \mathbb{R}^{d_{in}}$  is the output, then the general form of the recurrent neural network is defined as:

$$
\mathcal{M}_t = f(\mathcal{M}_{t-1}, x_t),
$$
 Write Operation (6)

$$
\mathbf{y}_t = g(\mathcal{M}_t, \mathbf{x}_t), \tag{7}
$$

where  $f(.,.)$  is the *read* and  $q(.,.)$  is the *write* corresponding functions. Note that here the subscript of  $\mathcal{M}_t$  shows the state of the memory at time  $t$ .

In this perspective, the recurrence formula of linear Transformers (see Equation 4) is equivalent to additively compress and write keys and values,  $(K_t, V_t)$ , into a matrix-valued memory unit  $\mathcal{M}_t$ . Therefore, when dealing with long context data, this additive nature of the process results in memory overflow, significantly damaging the performance of the model. To address this, studies have focused on two promising directions: (1) Adding forget mechanism: several studies have presented adaptive (data-dependent) forgetting gate mechanisms for linear models, where it can erase the memory when it is needed. As examples of such models, we refer to GLA (S. Yang, B. Wang, Shen, et al. 2024), LRU (Orvieto et al. 2023), Griffin (De et al. 2024), xLSTM (Beck et al. 2024), and Mamba2 (Dao and Gu 2024), which the later is also connected to the discretized version of traditional state space models (Gu and Dao 2024).(2) Improving the write operation: To overcome the additive nature of memory write operation in traditional recurrent models, Widrow and Hoff (1988) presented Delta Rule, in which before adding a memory (i.e., a pair of key and value), the model first removes its past value. To enhance the parallelizable training and scaling, S. Yang, B. Wang, Yu Zhang, et al. (2024) present a fast paralellizable algorithm. Finally, very recently, S. Yang, Kautz, and Hatamizadeh (2024) improved the DeltaNets by adding a forget gate.

**Memory Modules.** Memory has always been one of the core parts of the neural network designs (Graves, Wayne, and Danihelka 2014; JH Schmidhuber 1992; Jürgen Schmidhuber and Hochreiter 1997; J. Zhang et al. 2024). The idea of seeing linear layers as the key-value (associative) memory system backs to fast weight programs, in which dynamic fast programs are incorporated into recurrent neural networks to serve as writable memory (JH Schmidhuber 1992). The two learning rules of Hebbian (Hebb 2005) and delta (Prados and Kak 1989) are the most popular learning rules for fast weight programs, which have been extensively explored in various studies (Irie, Schlag, et al. 2021; Munkhdalai, Sordoni, et al. 2019; Munkhdalai and H. Yu 2017; Schlag, Irie, and Jürgen Schmidhuber 2021; JH Schmidhuber 1992; S. Yang, Kautz, and Hatamizadeh 2024; S. Yang, B. Wang, Yu Zhang, et al. 2024). All these models, however, are based on momentary surprise, missing the token flow in the sequences (see Section 3.1), and most of them lacks a forgetting gate, resulting in a poor memory management.

We further discuss the connection of our architectures with recent models in Appendix C. Additional related work are discussed in Appendix A.

{4}------------------------------------------------

### **Learning to Memorize at Test Time** $\mathbf{3}$

 $\overline{\mathcal{A}}$  o overcome the lack of long-term memory and to enable the model to learn, forget, and retrieve information, in this section, we present a neural long-term memory module, which is a meta models that learns to memorize at test time. In Section 3.1, we first discuss the motivation and the design of the neural memory. In Section 3.2, we discuss how our architecture design can benefit from a fast and parallelizable training. Finally, in Section 3.3, we augment our architecture using persistent memory module, in which we use learnable but data-independent parameters to learn meta information about the task.

#### **Long-term Memory** $3.1$

To design a neural long-term memory module, we need a model that can encode the abstraction of the past history into its parameters. An example of this can be LLMs that are shown to be memorizing their training data (Leybzon and Kervadec 2024; Schwarzschild et al. 2024; Staab et al. 2024). Therefore, a simple idea is to train a neural network and expect it to memorize its training data. Memorization, however, has almost always been known as an undesirable phenomena in neural networks as it limits the model generalization (Bayat et al. 2024), causes privacy concerns (Staab et al. 2024), and so results in poor performance at test time. Moreover, the memorization of the training data might not be helpful at test time, in which the data might be out-of-distribution. We argue that, we need an online meta-model that learns how to memorize/forget the data at test time. In this setup, the model is learning a function that is capable of memorization, but it is not overfitting to the training data, resulting in a better generalization at test time.

Learning Process and Surprise Metric. The key idea to train a long-term memory is to treat its training as an online learning problem, in which we aim to compress the past information  $x_1, \ldots, x_{t-1}$  into the parameters of our long-term neural memory module  $M_t$ . As discussed earlier, an event that violates the expectations (i.e., is surprising) is more memorable for humans (Mandler 2014). Inspired by this, a simple definition of surprise for a model can be its gradient with respect to the input. The larger the gradient is, the more different the input data is from the past data. Accordingly, using this surprise score, we can update the memory as:

$$
\mathcal{M}_t = \mathcal{M}_{t-1} - \theta_t \underbrace{\nabla \ell(\mathcal{M}_{t-1}; x_t)}_{\text{Surprise}}.
$$
 (8)

This surprise metric, however, can result in missing important information that comes after a big surprising moment. That is, the gradient can become extremely small after several surprising steps, leading to stocking in a flat area (i.e., local minima), and missing information about some parts of the sequence. From the human memory perspective, an event might not consistently surprise us through a long-period of time although it is memorable. The reason is that the initial moment is surprising enough to get our attention through a long time frame, leading to memorizing the entire time frame. To improve the above surprise metric (Equation 8), we break the surprise metric into  $(1)$  past surprise, which measures the surprise amount of a very recent past; and (2) momentary surprise, which measures the surprise of incoming data:

$$
\mathcal{M}_t = \mathcal{M}_{t-1} + S_t,\tag{9}
$$

$$
S_t = \eta_t \quad S_{t-1} \quad -\theta_t \quad \nabla \ell \left( M_{t-1}; x_t \right) \tag{10}
$$

Interestingly, this formulation is similar to gradient descent with momentum, where  $S_t$  is the momentum element. Therefore, the momentum here act as a memory of surprise across time (sequence length). In this formulation, the term  $\eta_t$  is a data-dependent surprise decay (a function of  $x_t$ ), controlling how surprise decays over time, and the term  $\theta_t$  is controlling how much of momentary surprise should be incorporated into the final surprise metric in a data-dependent manner. This data-dependency is particularly important in this design: While surprise of previous tokens might be needed to affect the surprise of the next token, it is mostly valid if all tokens are relevant and are in the same context. Accordingly, a data-dependent  $\eta$  can control if memory needs to: (1) ignore the last surprise by setting  $\eta_t \to 0$  (possibly due to the change of context), or (2) fully incorporate the last surprise by setting  $\eta_t \to 1$  (possibly as the token is highly relevant to its recent past tokens).

Objective. Our above surprise metric is based on a loss function  $\ell(.,.)$ , which is the objective that our memory is learning to act as it at test time. That is, our memory module is a meta model that learns a function based on the loss function  $l(:,.)$ . 

{5}------------------------------------------------

In this work, we focus on *associative memory*, in which we aim to store the past data as the pairs of keys and values. Given  $x_t$ , similar to Transformers (Vaswani et al. 2017), we use two linear layers to project  $x_t$  into a key and value:

$$
\mathbf{k}_t = x_t W_K, \qquad \qquad \mathbf{v}_t = x_t W_V, \tag{11}
$$

where  $W_K$  and  $W_V \in \mathbb{R}^{d_{in} \times d_{in}}$ . Next, we expect our memory module to learn the associations between keys and values. To this end, we define the loss as follows:

$$
\ell(\mathcal{M}_{t-1}; x_t) = \|\mathcal{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t\|_2^2
$$
(12)

By optimizing the above loss function in the inner-loop of our meta model (memory), the model learns how to memorize the mapping between keys and values at test time. Note that, similar to meta-learning models (Nichol 2018; Zintgraf et al. 2019), training of the memory is in the inner-loop, and so parameters  $W_K$  and  $W_V$  are hyperparameters in the above loss function. Accordingly, in the inner loop, we optimize  $\mathcal{M}$ 's weights, while in the outer-loop, we optimize other parameters of the entire architecture.

**Forgetting Mechanism.** When dealing with very large sequences (e.g., millions of tokens), it is crucial to manage which past information should be forgotten-even with a deep or a very large matrix-valued memory. To this end, we use an adaptive forgetting mechanism that allows the memory to forget the information that is not needed anymore, resulting in better managing the memory's limited capacity. That is, given the next token  $x_t$ , we modify the update rule as:

$$
\mathcal{M}_t = (1 - \alpha_t)\mathcal{M}_{t-1} + S_t,\tag{13}
$$

$$
S_t = \eta_t S_{t-1} - \theta_t \nabla \ell (M_{t-1}; x_t),
$$
(14)

where  $\alpha_t \in [0, 1]$  is the gating mechanism that flexibly controls the memory; i.e., decides how much information should be forgotten. For example, it can update the memory without affecting the past abstraction by letting  $\alpha_t \rightarrow 0$ , and can clear the entire memory by letting  $\alpha_t \to 1$ . Later in this section, we show that this weight decay mechanism is closely related to the gating mechanism in modern RNNs (Dao and Gu 2024; Orvieto et al. 2023).

**Memory Architecture.** In this paper, we focus on simple MLPs with  $L_M \ge 1$  layers as the architecture of our long-term memory. The main reason behind this choice is that we want to focus on better motivating the design of the long-term memory and ways that it can be incorporated into an architecture. However, our formulation and architectural design opens a new research direction to design neural architectures that are more effective and efficient in memorization of data. Recently, there has been a promising line of work to design such architectures (Berges et al. 2024; Cetin et al. 2024; J. Zhang et al. 2024), which incorporating them into our framework (i.e., replacing simple MLPs with such architectures) can be an interesting future work.

When using vector-valued or matrix-valued memory (De et al. 2024; Orvieto et al. 2023; S. Yang, B. Wang, Shen, et al. 2024), the memory module is compressing the past data and fit it into a line. That is, from the meta learning or online learning perspective (Yu Sun et al. 2024), using a matrix-valued memory  $\mathcal{M} = W \in \mathbb{R}^{d_{in} \times d_{in}}$  is equivalent to optimize  $\ell(W_{t-1}; x_t) = ||W_{t-1}k_t - v_t||_2^2$ , which is an online linear regression objective and so the optimal solution assumes the underlying dependency of historical data is linear. On the other hand, we argue that deep memory modules (i.e.,  $L_M \ge 2$ ). Aligning with the theoretical results that MLPs with at least two layers are strictly more expressive than linear models (Hornik, Stinchcombe, and White 1989), in Section 5.5, we show that deep memory modules are more effective in practice.

**Retrieving a Memory.** In the above, we discuss how one can design and train a long-term memory module that learns to memorize at test time. A key remaining question is: How one can retrieve information from the memory? We simply use the forward pass without weight update (i.e., inference) to retrieve a memory correspond to a query. Formally, given an input  $x_t$ , we use a linear layer  $W_Q$  to project the input, i.e.,  $q_t = x_t W_Q$  and retrieve the corresponding (or useful) information from the memory  $y_t$  by:

$$
y_t = \mathcal{M}^*(\mathbf{q}_t). \tag{15}
$$

{6}------------------------------------------------

![](_page_6_Figure_0.jpeg)

Figure 1: The illustration of how the training of neural memory can be done in parallel and using matmuls.

#### $3.2$ **How to Parallelize the Long-term Memory Training**

As discussed above, the design of our long-term memory module is equivalent to training a meta model by optimizing associative memory loss function  $\ell(\mathcal{M}_{t-1}; x_t) = ||\mathcal{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t||_2^2$  using gradient descent with momentum and weight decay. Therefore, in theory, the training of long-term memory module requires  $O(N)$  FLOPs, where N is the sequence length. However, in practice, we need to parallelize the training process and to fully take advantage of hardware accelerators (e.g., TPUs, GPUs), we need to tensorize the process and use more matmuls.

Next, we show that calculating the weights in the inner loop with mini-batch gradient descent, data-dependent learning rate, and weight decay can be reformulated so that it uses only matmuls and sum. We build upon the work of Yu Sun et al. (2024) that shows forward pass of a model optimizing with the mini-batch gradient descent (with constant learning rate) can be calculated using matmuls. We can split the sequence into chunks of size  $b \ge 1$ , and write the mini-batch gradient descent as:

$$
\mathcal{M}_t = (1 - \alpha_t)\mathcal{M}_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t) = \beta_t \mathcal{M}_0 - \sum_{i=1}^t \theta_i \frac{\beta_t}{\beta_i} \nabla \ell(\mathcal{M}_{t'}; x_i),
$$
(16)

where  $t' = t - \text{mod}(t, b)$ , and  $\beta_i = \prod_{i=1}^{i} (1 - \alpha_i)$ . For the sake of simplicity, we focus on the first chunk, i.e.,  $t = b$  and so  $t' = 0$ . Also, we explain the process for the case that  $M_t = W_t$  is linear. The process for MLPs with  $N_p \ge 2$  is similar. Using our loss function, we have:

$$
\nabla \ell(W_0; x_t) = (W_0 x_t - x_t) x_t^{\top} \Rightarrow \sum_{i=1}^b \theta_i \frac{\beta_b}{\beta_i} \nabla \ell(W_0; x_i) = \Theta_b \mathbf{B}_b (W_0 X - X) X^{\top}, \tag{17}
$$

where  $\Theta_b = \text{diag}([\theta_1 \quad \theta_2 \quad \dots \quad \theta_b])$  and  $\mathbf{B}_b$  is defined analogously on  $\frac{\beta_b}{\beta_i}$ s. Note that, we do not need to store all  $\Theta_{kb}$  and  $\mathbf{B}_{kb}$  for  $k = 1, \ldots, N/b$ , instead, we store these matrices for each chunk, resulting in using less memory. Next, we extend this representation so we can also incorporate the momentum term. In a chunk wise gradient descent with momentum, if we look at the momentum term, we have:

$$
S_t = \eta_t S_{t-1} - \theta_t u_t, \tag{18}
$$

where  $u_t = \nabla \ell (M_{t'}; x_t)$ . Note that, we can compute all  $u_t$  at the same time, and so Equation 18 is a linear recurrence with  $u_t$  as an input,  $S_t$  as the hidden state, and  $\eta_t$  as input-dependent transition value. Accordingly, we can use parallel associative scan (J. T. Smith, Warrington, and Linderman 2023) to calculate  $S_t$ s in this chunk.

Parameters as the Function of Chunks. Instead of making parameters like  $\alpha_t$ ,  $\theta_t$ , and  $\eta_t$  input-dependent (i.e., a function of token  $x_t$ ), we can make them functions of their chunk. Despite losing expressive power, this formulation can help to make the training even faster. In this case, we are using the same value for each of  $\alpha$ ,  $\theta$ , and  $\eta$  in each chunk. Accordingly, in Equation 17, we can store  $\Theta$  using a single scaler. Similarly we can make Equation 18 faster. That is, when  $\eta$  and  $\theta$  are learnable but time-invariant inside each chunk, this equation becomes a linear time-invariant system (LTI), which can be computed by a global convolution (Gu, Goel, and Re 2022). In our experiments, we make these parameters as the functions of tokens. However, such simplifications (i.e., as the function of chunks) can be the interest of future work to training larger models in more efficient manner.

{7}------------------------------------------------

![](_page_7_Figure_0.jpeg)

Figure 2: **Memory as a Context (MAC) Architecture.** This architecture includes three branches of (1) core, (2) contextual (long-term) memory, and (3) persistent memory. The core branch concatenates the *corresponding* long-term and persistent memories with the input sequence. Next, attention performs on the sequence and decides what part of the information should store in the long-term memory. At the test time, parameters corresponds to contextual memory are still learning, parameters corresponds to the core branch are responsible for in-context learning, and parameters of persistent memory are responsible to store the knowledge about tasks and so are fixed.

#### **Persistent Memory** $3.3$

Our long-term memory can also be seen as a contextual memory, meaning that the output is fully depend on the context. Therefore, in addition to our long-term memory, we also use a set of learnable but input-independent parameters to act as task-related memory. This type of memory has been referred to as persistent or meta-memory in the literature (X. Dong et al. 2024; Sukhbaatar, Grave, et al. 2019). Given  $N_p \ge 1$ , we use learnable parameters  $P = \begin{vmatrix} p_1 & p_2 & \dots & p_{N_p} \end{vmatrix}$  and append it to the start of our sequence: i.e., given a context window size of  $N$ , we modify the input as:

$$
x_{\text{new}} = \begin{bmatrix} p_1 & p_2 & \dots & p_{N_p} \end{bmatrix} || x, \tag{19}
$$

where  $\parallel$  is concatenation. Next, we discuss the motivation of persistent memory from three perspective:

Memory Perspective. As discussed earlier, our neural long-term memory is a contextual memory, in which all parameters are input-dependent. An effective memory system, however, also needs input-independent parameters to store the abstraction of the task knowledge. That is, mastering a task requires the memorization of the knowledge that how the task can be done, and these parameters are responsible for storing such knowledge.

**Feedforward Network Perspective.** In the Transformer architectures, there are fully connected layers after the attention module, which are shown to be similar to attention weights but with data-independent parameters. That is, Sukhbaatar, Grave, et al. (2019) showed that replacing the ReLU in fully connected layers with Softmax can results in an attention-like weights, in which weights are data-independent:

$$
FFN(x) = W_V \text{ Softmax} (W_K x). \tag{20}
$$

In fact,  $W_K$  and  $W_V$  are acting similar to K and V matrices in attention module when they are input-independent. The persistent memory weights are expected to have the same functionality, meaning that using them in the first part of the sequence leads to having input-independent attention weights (Sukhbaatar, Grave, et al. 2019).

**Technical Perspective.** Attention with causal mask has implicit bias toward initial tokens in the sequence, and so attention weights are almost always highly active for initial tokens, resulting in performance damage. From the technical perspective, these learnable parameters at the start of the sequence can mitigate such effect by redistributing the attention weights more effectively (Han et al. 2024; Xiao et al. 2024).

{8}------------------------------------------------

![](_page_8_Figure_0.jpeg)

(a) **Memory as a Context (MAC).** We segment the sequence and use full causal attention in each window. Again, the first  $N_p$  tokens are persistent memory and the next  $N_l$  are long-term memory tokens

![](_page_8_Figure_2.jpeg)

(b) **Memory as Gating (MAG).** We use sliding window attention (SWA) as a short-term memory and our neural memory module as a long-term memory, combining by a gating.

Figure 3: Attention masks for different variants of Titans.

#### $\mathbf{4}$ **How to Incorporate Memory?**

 $\mathfrak{g}_n$  important question that remained unanswered is: How one can effectively and efficiently incorporate the designed neural memory into a deep learning architecture? As discussed earlier, from a memory perspective, the pair of K and V matrices in transformers can be interpreted as an associative memory block. Due to their accurate modeling of dependencies and so their limited context window, we interpret them as short-term memory modules, attending to the *current* context window size. On the other hand, our neural memory with the ability to continuously learn from data and store it in its weights can play the role of a a long-term memory. In this section, we aim to answer the above question by proposing three different variants of Titans. Later in our experiments, we show that each of these variants has its own advantages/disadvantages and also can show a trade-off between the efficiency and effectiveness in very long-contexts.

## 4.1 Memory as a Context

In the first architecture design (see Figure 2), we treat the memory as a context to the current information. That is, given a long sequence  $x \in \mathbb{R}^{N \times d_{in}}$ , we first chunk the sequence into fixed-size segments  $S^{(i)}$  for  $i = 1, ..., N/C$ . Given the incoming segment  $S^{(t)}$ , we consider it as the current context and its past segment as the historical information. Therefore, let  $M_{t-1}$  be the state of long-term memory before segment  $S^{(t)}$ , we use the input context as the query to the memory  $M_{t-1}$  to retrieve the corresponding information from the long-term memory. That is, we retrieve the past information that corresponds to  $S^{(t)}$  as:

$$
h_t = \mathcal{M}_{t-1}^*(\mathbf{q}_t),\tag{21}
$$

where  $q_t = S^{(t)}W_Q$ . Next, we use this historical information along with our persistent memory parameters as the input sequence to the attention module:

$$
\tilde{S}^{(t)} = [p_1 \quad p_2 \quad \dots \quad p_{N_p}] \quad || \quad h_t || \quad S^{(t)}, \tag{22}
$$

$$
y_t = \text{Attn}\left(\tilde{\text{S}}^{(t)}\right). \tag{23}
$$

The structure of the attention map over the entire sequence is shown in Figure 3a. We then use  $y_t$  to update the long-term memory module for the next segment and the final output:

$$
\mathcal{M}_t = \mathcal{M}_{t-1}(y_t),\tag{24}
$$

$$
o_t = y_t \otimes \mathcal{M}_t^* (y_t). \tag{25}
$$

Note that, in the above, we are updating the weight of  $M_{t-1}$  through forward pass.

This architecture has two key advantages: (1) Attention by having both historical and current context, has the ability to decides whether given the current data, the long-term memory information is needed. (2) The attention module helps

{9}------------------------------------------------

![](_page_9_Figure_0.jpeg)

Figure 4: **Memory as a Gate (MAG) Architecture.** This architecture, similarly, has the three branches of (1) core, (2) contextual memory, and (3) persistent memory. It, however, incorporates only persistent memory into the context and combine memory with the core branch using a gating mechanism. At test time, the behavior is the same as Figure 2.

the long-term memory to store only useful information from the current context. That is, not all tokens in each segment are useful and memorizing all of them can result in memory overflow. Therefore, attention is helping the memory to understand which information is useful, better managing the memory capacity. (3) At test time: (i) persistent memory parameters are fixed as they encodes the knowledge about the task, which should not be changed; (ii) the attention module weights are in-context learner; and (iii) the long-term memory module is still learning (memorizing) the information at test time. That is, we update the weights of the neural memory even at test time as weights are encoding the abstraction of long past.

#### **Gated Memory** $4.2$

In the next variant (see Figure 4), in one branch, we directly use the input data to update the long-term memory, and in the second branch, we use a sliding window attention (SWA):

$$
\tilde{x} = \begin{bmatrix} p_1 & p_2 & \dots & p_{N_p} \end{bmatrix} || x, \tag{26}
$$

$$
y = SW-Attn^*(\tilde{x}), \tag{27}
$$

$$
p = y \otimes \mathcal{M}(\tilde{x}), \tag{28}
$$

where SW-Attn<sup>\*</sup> is sliding window attention with prefix (see Figure 3b). Note that, contrary to the previous design, we are not segmenting the input data. Also, we abuse the notation and use  $\mathcal{M}(x)$  to refer to the final output of the memory after all recursion over the tokens of the sequence. In the above equation,  $\otimes$  can be any non-linear gating. In our experiments, we normalize the outputs y and  $\mathcal{M}(\tilde{x})$  using learnable vector-valued weights, followed by a non-linearity  $\sigma(.)$ .

The overall attention mask of this design is shown in Figure 3b. In this design, sliding window attention is act as a precise short-term memory, while the neural memory module is acting as a fading memory for the model. This architecture design can also be seen as a multi-head architecture where the structure of heads are different (X. Dong et al. 2024).

#### **Memory as a Layer** $4.3$

The last variant uses the neural Memory As a Layer (MAL) of a deep neural network (see Figure 5). This architecture design is more common in the literature, where the hybrid models stack recurrent models with full or sliding window attentions. Given input  $x$ , we have:

 $\mathbf{1}$ 

$$
\tilde{x} = \begin{bmatrix} p_1 & p_2 & \dots & p_{N_p} \end{bmatrix} || x, \tag{29}
$$

$$
y = \mathcal{M}(x),\tag{30}
$$

 $\langle \alpha \alpha \rangle$ 

$$
o = \text{SW-Attn}(y), \tag{31}
$$

{10}------------------------------------------------

![](_page_10_Figure_0.jpeg)

Figure 5: **Memory as a Layer (MAL) Architecture.** In this architecture, the memory layer is responsible to compress the past and current context before the attention module.

where SW-Attn is sliding window attention. The main drawback of this design is that the power of the model is limited by each of the layers and so it cannot take advantage of the complementary data processing of attention and neural memory module. In our experiments, for evaluating memory in this design, we use a similar architecture as H3 (D. Y. Fu et al. 2023), where we replace the the sequence model with our neural memory module (LMM).

**Memory Without Attention.** Although in the above, we discussed MAL as the combination of LMMs and attention in a sequential manner, one simple variant of MAL is to treat LMM as a sequence model without any attention. From the memory perspective, as discussed in Section 1, we expect each part of the memory system to work independently, even if other components are disturbed. Therefore, a long-term memory module should still be a powerful model even without short-term memory (i.e., attention). We refer to this variant as LMM or Titans (LMM) in our experiments. We provide additional discussions on the connection of Titans and other modern recurrent models in Appendix C.

## 4.4 Architectural Details

For the sake of simplicity and presentation, we avoid discussing the implementation details like using residual connection, gating with linear layer, and normalization. In all blocks, we use residual connections. In our implementation, we use SiLU(.) activation (Elfwing, Uchibe, and Doya 2018) as the non-linear activation for computing query, key, and values and normalize queries and keys using  $\ell_2$ -norm.

Convolution. Following the recent modern linear recurrent models (Gu and Dao 2024; S. Yang, Kautz, and Hatamizadeh 2024), we incorporate a 1D depthwise-separable convolution layer after each of the query, key, and value projections. While not significantly affect the performance, these 1D convolutions have shown performance improvement and are also computationally efficient.

Gating. We also follow the recent architectures that use normalization and gating with a linear layer before the final output projection (Mehta et al. 2023).

**Theorem 4.1.** Contrary to Transformers, diagonal linear recurrent models, and DeltaNet, all of which are limited to  $TC^0$  (Merrill, Petty, and Sabharwal 2024), Titans are capable of solving problems beyond  $TC^0$ , meaning that Titans are theoretically more expressive than Transformers and most modern linear recurrent models in state tracking tasks.

### $\mathbf{5}$ **Experiments**

![](_page_10_Picture_10.jpeg)

ext, we evaluate the performance of Titans and its variants in language modeling, commonsense reasoning, needle in haystack, DNA modeling, and time series forecasting tasks<sup>1</sup>. In more details, in this section, we answer the following empirical questions: (1) How do Titans perform compared to baselines in downstream tasks? (see §5.2,

<sup>&</sup>lt;sup>1</sup>In the first version of the work, we aim to provide insights/evidences about why the learning paradigms of Titans are effective. We are working on finalizing the results of larger models and will report them in the next version.
      