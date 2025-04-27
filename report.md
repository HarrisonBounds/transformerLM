# Decoder Based Transformer - Homework 1

### Modification to original code

The original code for this homework was an Encoder-Decoder Transformer architecture that had a source and target vocabulary while also using cross-attention. To alter this into a decoder only autoregressive language model, the first step was to get rid of the encoder completely. I commented out the original encoder and changed the input paramteres to the decoder to exclude the output of the encoder (). Second, the cross-attention is no longer needed. Self-attention is the only thing this model needs to utilize. For this reason, the lines that performed cross attention in the `MultiHeadAttention` class were commented out. 
---

### Results 

---

### Hyperparameters

---

### Training Script and weights

---

### Training on Wikitext-103

---

### Replaced Attention with Euclidean distance: Results

