# Homework 2 - Natural Language Processing

### Classification using Pretrain BERT model

This code implements a fine-tuning approach for the BERT-base-uncased model for a multiple-choice question answering task. The goal is to predict the correct answer from four choices given a fact and a question stem. Below is a desccription of what is going on:

1.  **Data Preparation and Encoding:**
    * The input data is assumed to be in JSONL format, where each line represents a question instance with fields like `"fact1"`, `"question"` (containing `"stem"` and a list of `"choices"` with `"text"` and `"label"`), and `"answerKey"`.
    * The `MultipleChoiceDataset` class handles the loading and preprocessing of this data. For each question, it constructs four input sequences, one for each choice, in the following format: `[CLS] <fact> <stem> <text of choice #i> [SEP]`.
    * The `BertTokenizer` from the `bert-base-uncased` checkpoint is used to tokenize these sequences. Special tokens (`[CLS]` and `[SEP]`) are added, and the sequences are padded or truncated to a `max_length` (set to 128 in this code). Attention masks are also generated to indicate the actual tokens versus padding tokens.
    * The correct answer label (A, B, C, or D) is converted to a numerical index (0, 1, 2, or 3) using a `label_map`.

2.  **Model Architecture:**
    * The `BertForMultipleChoiceCustom` class defines the model. It utilizes the pre-trained `BertModel` from the `bert-base-uncased` checkpoint as the base encoder.
    * A single linear layer (`nn.Linear`) is added on top of the hidden state of the `[CLS]` token from the final layer of the BERT model. This linear layer maps the `[CLS]` embedding to a single output value (logit) for each choice.

3.  **Classification Methodology:**
    * During the forward pass, for each input instance (with four choices), the four encoded sequences are passed through the `BertModel`.
    * The hidden state corresponding to the `[CLS]` token (the first token) from the last layer is extracted for each of the four choices. This `[CLS]` embedding is intended to represent the entire input sequence's contextualized information.
    * Each of these four `[CLS]` embeddings is then fed into the linear classification layer, producing a single logit score for each choice.
    * These logits are then used with a `CrossEntropyLoss` function during training to compare against the true label. The model learns to assign a higher logit score to the correct choice.

4.  **Fine-tuning:**
    * The `BertForMultipleChoiceCustom` model is fine-tuned on the training dataset.
    * The `AdamW` optimizer (from `torch.optim`) is used to update the model's parameters. A learning rate of 2e-5 is set.
    * The training loop iterates for a specified number of epochs (3 in this code). In each epoch, the model is trained on batches of data. The loss is calculated, backpropagation is performed, and the optimizer updates the weights.
    * After each epoch, the model's performance is evaluated on the validation dataset to monitor progress and potentially tune hyperparameters (though not explicitly done in this basic script). Accuracy is calculated as the percentage of correctly predicted answers.

5.  **Inference:**
    * For inference on the validation and test sets, the same data preprocessing steps are applied to create the input sequences.
    * The fine-tuned model is used to predict the logits for each of the four choices.
    * The predicted answer is the choice with the highest logit score (obtained using `torch.argmax` along the choice dimension).
    * The accuracy is then calculated by comparing the predicted answers with the true labels.

6.  **Zero-shot Evaluation:**
    * To assess the baseline performance of the pre-trained `bert-base-uncased` model without any task-specific fine-tuning, a separate evaluation is performed on the validation and test sets using a newly initialized `BertForMultipleChoiceCustom` model with the pre-trained weights. The accuracy obtained in this step is the zero-shot accuracy.

## Reported Accuracies

*(Note: The actual zero-shot and fine-tuned accuracies will vary depending on the dataset, hyperparameters, and training run. You will need to run the code on your data to obtain these values. The following is a placeholder for where you would report your results.)*

**Zero-Shot Accuracy:**

* **Validation Set:** \[Insert your zero-shot validation accuracy here]%
* **Test Set:** \[Insert your zero-shot test accuracy here]%

**Fine-tuned Accuracy:**

* **Validation Set:** \[Insert your best fine-tuned validation accuracy here]%
* **Test Set:** \[Insert your fine-tuned test accuracy here]%

## Limitations of the Approach and Possible Solutions

1.  **Input Length Constraints:** BERT has a maximum input sequence length (typically 512 tokens). If the combined length of the fact, stem, and choice text exceeds this limit, the tokenizer will truncate the input, potentially leading to a loss of crucial information.
    * **Possible Solutions:** Explore different truncation strategies (e.g., prioritizing the choice text), or if the context is very long, consider techniques like sliding window (though this adds complexity to the classification).

2.  **Information Aggregation at `[CLS]`:** Relying solely on the `[CLS]` token's final hidden state to represent the entire input sequence for classification might not capture all the relevant information distributed across the sequence.
    * **Possible Solutions:** While the prompt specifies using the `[CLS]` token, alternative aggregation methods like averaging the hidden states of all tokens or using a pooling layer could be explored in future iterations (though this deviates from the current approach).

3.  **Semantic Similarity Focus:** The linear layer on top of the `[CLS]` embedding primarily learns to map the semantic representation to a score. It might struggle with tasks requiring more complex logical reasoning or understanding of relationships beyond surface-level similarity.
    * **Possible Solutions:** The fine-tuning process should ideally teach the model to perform the necessary reasoning based on the task. However, if performance is limited, exploring more sophisticated classification layers or incorporating external knowledge might be considered in more advanced approaches.

4.  **Computational Cost:** Fine-tuning large transformer models like BERT can be computationally expensive and time-consuming, especially with larger datasets and more training epochs.
    * **Possible Solutions:** Utilize GPUs for faster training, experiment with smaller batch sizes if memory is a constraint, or explore more efficient fine-tuning techniques if the task allows.

5.  **Hyperparameter Tuning:** The hyperparameters (learning rate, batch size, number of epochs, maximum sequence length) are set to fixed values in this script. These values might not be optimal for the specific dataset.
    * **Possible Solutions:** Implement a hyperparameter search strategy (e.g., grid search or random search) using the validation set to find a better configuration.

This approach provides a basic yet effective method for fine-tuning BERT for multiple-choice question answering by treating each choice as a separate input and using the `[CLS]` token embedding for classification. The reported accuracies will indicate the effectiveness of this strategy on the given dataset.