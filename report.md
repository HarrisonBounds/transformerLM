# Decoder Based Transformer - Homework 1

### Modification to original code

The original code for this homework was an Encoder-Decoder Transformer architecture that had a source and target vocabulary while also using cross-attention. To alter this into a decoder only autoregressive language model, the first step was to get rid of the encoder completely. I commented out the original encoder and changed the input paramteres to the decoder to exclude the output of the encoder (). Second, the cross-attention is no longer needed. Self-attention is the only thing this model needs to utilize. For this reason, the lines that performed cross attention using the MultiheadAttention class in the `DecoderLayer` class were commented out (). After this, a single vocabulary size was taken from the corpus gathered from the tokenizer, replacing the need for a target and source vocab. The vocab was then tokenized, embedded, then shifted to get the input and prediction tensors. Next, the model needed to be able to handle any batch sizes for optimal training, so I created a data_generator class that generates the data in batches with padding. This function returns the input and target sequences that are passed into the transformer architecture. 

Finally, the training, validation, and test functions were created to train the decoder transformer model. The `train_model()` function orchestrates the training loop, processing batches of tokenized text with gradient updates, logging loss and perplexity, and periodically saving model checkpoints, while also calling `validate_model()` and `test_model()` after each epoch. These evaluation functions share a similar structure: they disable gradient computation (`torch.no_grad()`), compute masked autoregressive predictions using the same `nopeak_mask` as training, and calculate cross-entropy loss (ignoring padding tokens) and perplexity, but differ in their data sources (validation or test sets) and batch iteration handling. All three functions log epoch-wise metrics (loss and perplexity) to both console and a file, enabling performance tracking across the training lifecycle. The unified design ensures consistent behavior across training and evaluation while maintaining separation between training dynamics (optimizer steps, gradient clipping) and inference-mode evaluation.

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

