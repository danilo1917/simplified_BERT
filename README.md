# Reproducing Transformer-Based Architectures: Mini-BERT Implementation
## Danilo Freitas Vieira
## Ian Otoni Vieira Gomes

# Introduction
The motivation for this project is rooted in the influential paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2019). The paper introduced the Bidirectional Encoder Representations from Transformers (BERT), a language representation model that achieved state-of-the-art results across multiple NLP tasks. By utilizing a bidirectional Transformer architecture and novel pre-training objectives—Masked Language Model (MLM) and Next Sentence Prediction (NSP)—BERT successfully captures contextual information from both left and right directions in a sentence.
BERT's results established new performance benchmarks in tasks such as the General Language Understanding Evaluation (GLUE) benchmark, SQuAD for question answering, and SWAG for common sense inference. However, the original BERT model requires extensive computational resources for pre-training and fine-tuning, making it difficult to reproduce on modest hardware.
In this project, we implemented a simplified version of BERT, named Mini-BERT, to perform binary text classification on the IMDB sentiment analysis dataset. The IMDB dataset, which contains 50,000 movie reviews labeled as positive or negative, was chosen due to its smaller size and accessibility for training within resource constraints.

## IMDB Dataset
You can find the IMDB dataset for Binary Sentiment Analysis at https://huggingface.co/datasets/stanfordnlp/imdb/viewer/plain_text/test

# Implementation Details
The implementation of Mini-BERT preserves the core components of the original BERT model but simplifies the architecture to suit limited computational resources. The model consists of an embedding layer, multiple Transformer encoder layers, and a final classification layer. Each component follows the structure of a Transformer as described in the original BERT paper.
Overall Structure of the Code
The embedding layer generates token and position embeddings, ensuring that input sequences retain both semantic and positional information.
The multi-head self-attention mechanism enables the model to learn relationships between words, regardless of their distance within the input sequence.
Each Transformer encoder layer combines self-attention with a position-wise feed-forward network and applies layer normalization to stabilize training.
The final classification layer processes the representation of the [CLS] token to output predictions for binary sentiment classification.
Differences from the Original BERT
Model Size: The number of hidden units, intermediate dimensions, and attention heads were reduced.
Simplified Objective: The NSP task was removed as it is not necessary for single-sentence classification tasks.
Dataset Scope: Instead of pre-training on large datasets like Wikipedia or BooksCorpus, we directly fine-tuned the model on the IMDB dataset.
Manual Implementation: Key Transformer components, such as self-attention and feed-forward layers, were implemented manually to gain a deeper understanding of the architecture.

# Experimental Setup
The experimental setup was designed to train and evaluate the Mini-BERT model for binary sentiment classification. The key details are as follows:
Model Architecture: Mini-BERT with 12 Transformer encoder layers, 12 attention heads, and a hidden size of 192.
Optimizer: Adam optimizer with a learning rate of 5e-5.
Batch Size: 32.
Epochs: 15 epochs for training.
Dataset: IMDB dataset with input sequences truncated or padded to a maximum length of 128 tokens.
Hardware: Google Colab environment with a GPU (T4). 
Differences from the Original Setup
Unlike the original BERT paper, which involves pre-training on massive datasets followed by fine-tuning, this project directly fine-tuned Mini-BERT on the IMDB dataset. This decision was made to accommodate hardware limitations and time constraints. The reduced model size and simplified training loop also distinguish this setup from the full-scale BERT implementation.

# Results
The Mini-BERT model was trained and evaluated on the IMDB dataset, with accuracy as the primary performance metric.
## Discussion of Results
The Mini-BERT model achieved a final accuracy of 75.20%. The test set comprises 25,000 sentences, evenly distributed between 12,500 positive and 12,500 negative reviews.
![image](https://github.com/user-attachments/assets/26fb109c-cef3-4441-8cd9-b393b08fa23b)

![image](https://github.com/user-attachments/assets/bd021629-c5c7-4da7-b062-d40aa7fb640d)

![image](https://github.com/user-attachments/assets/979e2d2f-378a-4ed5-be19-983c963cb755)

# Discussion
Reproducing a Transformer-based model like BERT presented several challenges, primarily due to hardware limitations and dataset size. The original BERT model requires large-scale pre-training using datasets like Wikipedia and BooksCorpus, followed by fine-tuning on specific tasks. This process is computationally intensive and hard to do on platforms like Google Colab. Also, we faced challenges with matrix reshaping, because ensuring the correct reshaping and transposing of the Q, K, and V matrices was non-trivial. The input tensor must be split into multiple attention heads while preserving batch size and sequence length. Misunderstanding these operations led to misaligned tensor shapes and incorrect attention scores some times. In conclusion, developing this project provided a valuable and enriching experience. It allowed for a deeper understanding of Transformer architectures, particularly the intricacies of self-attention mechanisms and model implementation. Despite the challenges faced, the process offered significant insights into model design, dataset processing, and performance evaluation. This hands-on experience reinforced both theoretical knowledge and practical skills, making it a rewarding learning opportunity.

# Running our code
You just need to execute our colab, cell by cell.
