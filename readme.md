# Design Decisions

## Memory Module 
- retrieve(q: [B, S, D]) -> [B, S, D]
- update(x: [B, S, D]) -> None

todo: finalize this

## MACAttention 
[B, S, D] -> [B, S, D]


## MACTransformer

stack MACBlocks and sprinkle in some sw attention ?

## Should MemoryModules Be Shared? 

lucidrains doesn't share, but could be interesting to have global memory. 

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

ok this is minor. but i like B better. 

Transformer 
- input embedding
- absolute positional embedding 
- maybe intersegement embedding

MacAttention 
- equations 21-25




## How to treat persistent memory 

### why not treat persistent memory as tokens directly in the sequence like we do with long term memory? 
I'm thinking, persistent tokens are just attention sinks. https://gemini.google.com/app/4e0fc507da3d8fcf

They’re position‑agnostic anchors

RoPE is applied to (q, k) before concatenating pmk, pmv, so persistent keys do not get rotary positions in this implementation. They become global, position‑free anchors that every segment can attend to the same way. If you simply prepended tokens, they would get positional encoding and drift with segment layout.

### Should persistent memory be shared across layers? 
I think not. If we keep the same persistent memory throughout all blocks, we may get overload the persistent memory unless we have some sort of forget gate. 
TODO: let's experiment with this

## How to implement equation 25? 
The paper doesn't specify what ⊗ is. Let's use an mlp to mix memory output back 


## TODO: Segmented Attention
Paper says to segment sequence. each segment is added as a batch 



## Positional Embeddings
Lets start with standard absoulte positional embeddings. generate position_ids inside transformer.forward. then we can decide whether we want inter-segment embeddings as well once we get to that. apply rope within macattention



what conepts do we have: 
- long term memory 
- persistent memory 
- macattention 

macblock calls neural memory. which returns long term memory tokens
prepend to seq and pass to macattention 
output of macattention is used to nueralmemory.update

return neuralmemory.retrieve(out) + out  eq.25



TODO LIST: 

- Implement Mactransformer without segmentation
    -   keep current llm generated macattention.
    -   apply absolute pos embedding in forward ✅
    -   loop through blocks ✅
    -   output projection to vocab space ✅
    -   return logits ✅