#  Tweet Classification using LSTM and Bidirectional LSTM

This repository contains an end-to-end Deep Learning pipeline built with **Keras** and **TensorFlow** to classify tweets/text data into binary categories. It implements text preprocessing, tokenization, sequence padding, and compares the performance of a standard **LSTM** model against a **Bidirectional LSTM** model.

##  Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline & Architecture](#pipeline--architecture)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [License](#license)

---

##  Project Overview
Social media text data is highly unstructured, noisy, and full of abbreviations, emojis, and slang. This project demonstrates how to pre-process raw tweets using **NLTK**, vectorize them using Keras's **Tokenizer**, and build recurrent architectures (LSTM & BiLSTM) capable of capturing sequence order and contextual relationships in text.

##  Dataset
The dataset used is `mydataset.csv`, which consists of two columns:
- `Labels`: Binary indicator (0 or 1)
- `Data`: Raw tweet messages

##  Pipeline & Architecture

### 1. Preprocessing
- **Lowercasing**: Standardizing all characters to lowercase.
- **Digit Removal**: Stripping out numerical values.
- **Punctuation Removal**: Eliminating non-alphanumeric punctuation symbols.
- **Stopwords Removal**: Removing common filler words using NLTK's English stopword corpus.

### 2. Feature Engineering
- **Tokenization**: Converting words into unique integer tokens limiting the vocabulary size to the top 5,000 most frequent words.
- **Sequence Padding**: Padding or truncating sequences to a fixed length of 150 tokens to ensure uniform input shape.

### 3. Deep Learning Architectures
- **Embedding Layer**: Learns dense, low-dimensional vector representations for words.
- **Standard LSTM**: Processes text sequentially in a single direction.
- **Bidirectional LSTM**: Processes sequences in both forward and backward directions to capture deeper context.
- **Dense / Output Layer**: Uses a sigmoid activation function for binary classification.

---

##  Installation & Requirements
To run the notebook or script locally, clone this repository and install the required dependencies:

```bash
pip install pandas nltk tensorflow scikit-learn
```

Make sure to also download the required NLTK corpuses within your Python shell:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

##  Usage
1. Ensure your dataset is named `mydataset.csv` and placed in the root directory.
2. Run the notebook cells sequentially to clean the text, build the vocab, and train both neural networks.
3. The notebook will automatically compile metrics and display the test evaluation output.

---

##  Model Performance
After training both models for 5 epochs on a train-test split ratio of 80:20, the final validation metrics are:

| Model | Training Accuracy | Validation Accuracy | Loss |
| :--- | :---: | :---: | :---: |
| **Standard LSTM** | ~96.42% | ~89.16% | Binary Crossentropy |
| **Bidirectional LSTM** | ~96.19% | ~88.70% | Binary Crossentropy |

Both architectures yield robust generalization results above 88% on validation data. Feel free to adjust sequence lengths, embedding dimensions, or regularization rates to further improve model performance.
