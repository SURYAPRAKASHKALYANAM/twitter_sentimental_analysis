# Twitter Sentiment Analysis

## Introduction

This repository contains code for performing sentiment analysis on Twitter data using various machine learning models and natural language processing techniques, including TF-IDF, GloVe, and Word2Vec. The goal is to classify tweets into different sentiment categories such as positive, negative, or neutral.

## Prerequisites

Before you can run the code in this repository, make sure you have the following prerequisites installed:

- Python 
- NumPy
- Pandas
- Scikit-Learn
- XGBoost
- NLTK
- Gensim

You can install these dependencies using `pip`:

📋 Install the required Python packages as mentioned in the prerequisites section.

## Data

To perform sentiment analysis, you'll need a dataset of Twitter tweets with labeled sentiments (positive, negative). You can either collect your own dataset or use publicly available datasets. 

# Preprocessing Steps in NLP
1. **Handling Null Values (isnull)**:
   - **Description**: Address missing or null values in the text data.
   - **Purpose**: Avoid errors during analysis caused by missing data. 🧐
     
2. **Removing Emojis (convert_emojis)**:
   - **Description**: Replace or remove emojis from the text.
   - **Purpose**: Prevent emojis from affecting sentiment analysis. 😃
     
3. **Lowercasing (lower)**:
   - **Description**: Convert all text to lowercase to ensure uniformity.
   - **Purpose**: Standardize text, treating words in a case-insensitive manner. 🔤

4. **Removing Punctuation (remove_punct)**:
   - **Description**: Eliminate punctuation marks from the text.
   - **Purpose**: Focus on words by removing non-alphanumeric characters. 🚫

5. **Tokenization (word_tokenize)**:
   - **Description**: Split the text into individual words or tokens.
   - **Purpose**: Divide text into manageable units for analysis. ✂️

6. **Removing Whitespaces (whites)**:
   - **Description**: Strip extra whitespace characters.
   - **Purpose**: Standardize text formatting by removing unnecessary spaces. ⌨️

7. **Stopword Removal (stopwords)**:
   - **Description**: Eliminate common words (e.g., "the," "and," "is").
   - **Purpose**: Reduce dimensionality and focus on content words. 🛑

8. **Lemmatization (WordNetLemmatizer)**:
   - **Description**: Reduce words to their base or dictionary form.
   - **Purpose**: Group similar words and simplify analysis. 📚

9. **Stemming (PorterStemmer)**:
   - **Description**: Reduce words to their root form.
   - **Purpose**: Further text simplification and feature reduction. 🌱


# Word Embedding Techniques

## TF-IDF (Term Frequency-Inverse Document Frequency) 📊

- **Description**: TF-IDF is a technique used in natural language processing and information retrieval. It assigns a numerical value to each word in a document based on how often it appears in the document (term frequency) and how unique it is across a collection of documents (inverse document frequency).

- **Purpose**: TF-IDF is primarily used for text-based tasks like document retrieval, text classification, and information retrieval. It helps identify words that are both common in a document and unique across the corpus, making them good indicators of the document's content.

## Word2Vec (Word to Vector) 💡

- **Description**: Word2Vec is a word embedding technique that learns vector representations of words by training a shallow neural network on a large corpus of text. It maps words to continuous vector spaces where words with similar meanings have similar vector representations.

- **Purpose**: Word2Vec is widely used for capturing semantic relationships between words. It's valuable for natural language understanding tasks like word analogy, document clustering, and sentiment analysis.

## GloVe (Global Vectors for Word Representation) 🌐

- **Description**: GloVe is another word embedding technique that leverages global word-word co-occurrence statistics from a large corpus to create word vectors. It aims to capture not only semantic similarity but also syntactic patterns in the text.

- **Purpose**: GloVe is useful for tasks that require a deeper understanding of both semantic and syntactic relationships between words. It's applied in various NLP tasks, including machine translation, sentiment analysis, and text summarization.

These techniques are essential for converting text data into numerical vectors, allowing machine learning models to process and understand textual information effectively.


## Models

We've implemented several machine learning models for sentiment analysis:

- StackingClassifier
- MLPClassifier (Multi-layer Perceptron)
- XGBoost
- GaussianNB
- KNeighborsClassifier
- RandomForestClassifier
- DecisionTreeClassifier
- SVC (Support Vector Classifier)
- LogisticRegression

These models are trained on the features extracted from the text data using the TF-IDF, GloVe, and Word2Vec methods.
