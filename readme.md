# Summary 

This repository is my attempt to implement Titans: Learning to Memorize at Test Time. 
https://arxiv.org/abs/2501.00663

The main ideological predecessor to this paper is Learning to Learn at Test Time (https://arxiv.org/abs/2407.04620) which describes test-time-training layers. The idea is that the outer optimization loop (next token prediction) is responsible for training most parameters. However, during a single forward pass, another inner optimization loop optimizes a smaller set of parameters to learn as the model reads a sequence. 

Titans extends TTT by introducing momentum and forget gates to the memory architecture as well as presenting three ways to integrate memory into a vanilla transformer (memory as context, memory as gate, and memory as layer). 

This repo mainly focusses on memory as context. 

# Design Decisions

## Main Separation of Concerns

### Memory Module 
- contains MLP whose weights are updated by inner loops in forward pass
- retrieve(q: [B, S, D]) -> [B, S, D]
- store(x: [B, S, D]) -> None
### MACTransformer
- embeddings
    - positional 
    - token 
- stacks of MACBlocks
    - prepend longterm mems  
    - project out persistent mems
    - mac (equations 21-25)

## How Should MACBLocks interact with attention 

they didn't really talk about this in the paper. they just say: 

s = persistent + longterm + seq
y = attn(s)
update memory with y 
then use updated memory to combine with y 


but technically how to do this: 

### Option A
transformer forward:
    prepend long term memory 

call MacAttention on resulting sequence: 
    proj out qkv with longterm + seq
    don't actually store persistent tokens in model space. store them in projected space
    prepend pmk and pmv to k and v
    scaled_dot_product_attention 

### Option B 
prepend long term in macattention forward. 



## How to treat longerm memory 

### is long term memory block level or transformer level? 
long term memory lives in sequence space. we know that for sure. 

lets think of memory mlps as little storage units. 
if we share those storage units across layers, they might be storing information in different vector spaces. 
so maybe it makes sense to keep long term memory at the block level. it's also much simpler to implement this way. 
this paper is about accumulating information across the sequence dimension. let's not try to innovate on information flow across layers

I think a future todo could be to implement some sort of transformer level long term memory. 


## How to treat persistent memory 

### why not treat persistent memory as tokens directly in the sequence like we do with long term memory? 
I'm thinking, persistent tokens are just attention sinks. https://gemini.google.com/app/4e0fc507da3d8fcf

They’re position‑agnostic anchors

RoPE is applied to (q, k) before concatenating pmk, pmv, so persistent keys do not get rotary positions in this implementation. They become global, position‑free anchors that every segment can attend to the same way. If you simply prepended tokens, they would get positional encoding and drift with segment layout.

### Is persistent memory block level or tranformer level? 
I think persistent memory is block level. they are just attention sinks

## How to implement equation 25? 
The paper doesn't specify what ⊗ is. Let's use an mlp to mix memory output back 


## Positional Embeddings
Lets start with standard absoulte positional embeddings. generate position_ids inside transformer.forward. then we can decide whether we want inter-segment embeddings as well once we get to that. apply rope within macattention


## Memory 
### Store
- surprise is calculated per mb of sequence 
- currently implemented in a for loop.
- future: 
    - associative scan for linear recurrences: https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf
    - understanding scans: https://chatgpt.com/c/689902a0-11cc-8333-9268-34d8c7b1e2e2

# Todo
- persistent memory 
- scans
- windowed attention
- forget gates 


