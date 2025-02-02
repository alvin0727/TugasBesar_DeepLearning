# Fake News Detection Using Word2Vec and LSTM

This project aims to detect fake news using Word2Vec for text representation and LSTM (Long Short-Term Memory) for classification. The project is developed using Python and machine learning libraries such as TensorFlow/Keras, Gensim, and Scikit-learn.

## Table of Contents

- [Project Description](#project-description)
- [Preprocessing](#preprocessing)
- [Dataset Structure](#dataset-structure)
- [Model Architecture](#model-architecture)
- [Evaluation Results](#evaluation-results)
- [GUI](#gui)

## Project Description

This project utilizes Natural Language Processing (NLP) techniques to distinguish between real and fake news. The key steps include:
1. **Text Preprocessing**: Cleaning and preparing text data.
2. **Word2Vec**: Generating word embeddings for text representation.
3. **LSTM**: Building a deep learning model for text classification.
4. **Evaluation**: Measuring model performance using metrics such as accuracy, precision, recall, and F1-score.

## Preprocessing

The preprocessing phase involves cleaning and preparing text data before training the model. The steps include:

1. **Contraction Expansion**:
   - Expanding contractions such as "don't" to "do not" using the `contractions` library.

2. **Text Cleaning**:
   - Removing URLs, mentions (@), and hashtags (#).
   - Removing special characters (except letters and numbers).
   - Converting text to lowercase.
   - Separating numbers from text (e.g., "123abc" to "123 abc").

3. **Tokenization**:
   - Splitting text into tokens (words) using `RegexpTokenizer`.

4. **POS Tag Conversion to WordNet**:
   - Converting POS tags from NLTK to WordNet format for lemmatization.

5. **Lemmatization**:
   - Converting words to their base form using `WordNetLemmatizer` with POS tagging.

6. **Stopword Removal**:
   - Removing common words (stopwords) that do not provide significant meaning.

7. **Short Word Removal**:
   - Removing words shorter than a specified minimum length (default: 3 characters).

8. **Preprocessing Pipeline**:
   - Combining all preprocessing steps into a single pipeline for automated text processing.

## Dataset Structure
The dataset is sourced from the **ISOT**:  
[Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)

The dataset used in this project must follow this format:
- **Title**: News title.
- **Text**: News content.
- **Label**: `1` for fake news, `0` for real news.

Example dataset structure:

| Title                     | Text                          | Label |
|---------------------------|-------------------------------|-------|
| As U.S. budget fight looms, Republicans flip their fiscal scrip   | WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress... | 0     |
| Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing  | Donald Trump just couldn t wish all Americans a Happy New Year... | 1     |

## Model Architecture

1. **Word2Vec**:
   - Used to convert words into vectors.
   - Embedding dimension: 100-300.
   - Embedding weights are not updated during training.

2. **LSTM**:
   - Two LSTM layers with 128 units in the first layer and 64 units in the second layer.
   - The first LSTM layer returns the full sequence to the next layer.
   - The second LSTM layer returns only the final output.

3. **Regularization**:
   - Dropout applied after each LSTM and Dense layer to prevent overfitting.
   - Dropout rates: 0.3 after the first LSTM, 0.5 after the second LSTM, and 0.2 after the Dense layer.

4. **Dense Layers**:
   - A Dense layer with 32 units and ReLU activation before the output layer.
   - The output layer uses a single neuron with sigmoid activation for binary classification.

5. **Optimizer**:
   - Adam optimizer with an adjustable learning rate.

6. **Loss Function**:
   - Binary Crossentropy is used as the loss function.

7. **Hyperparameter Tuning with Optuna**:
   - Used to automatically search for the best hyperparameter combinations.
   - Optimizes LSTM units, learning rate, batch size, and dropout rate.

## Evaluation Results

The following are the evaluation results of the model on the given dataset:
- **Test Accuracy**: 0.9624
- **Test F1-Score**: 0.9646

### Confusion Matrix:

|                   | Predicted Negative | Predicted Positive |
|-------------------|-------------------|-------------------|
| **Actual Negative** | 4039              | 208               |
| **Actual Positive** | 130               | 4603              |

![Evaluation](https://github.com/alvin0727/TugasBesar_DeepLearning/blob/main/Images/image1.png)
![Evaluation](https://github.com/alvin0727/TugasBesar_DeepLearning/blob/main/Images/image2.png)


## Gui

![Gui](https://github.com/alvin0727/TugasBesar_DeepLearning/blob/main/Images/image3.png)

We appreciate your interest in this project. Thank you for taking the time to explore our work!