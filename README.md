# 📖 Natural Language Processing Specialization

[![Course](https://img.shields.io/badge/DeepLearning.AI-Course-blue)](https://www.coursera.org/specializations/deep-learning)
[![Platform](https://img.shields.io/badge/Platform-Coursera-blueviolet)](https://www.coursera.org)
[![Language](https://img.shields.io/badge/Language-Python-green)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)](https://www.tensorflow.org/)

This repository contains resources, assignments, and projects from the [**Natural Language Processing Specialization**](https://www.coursera.org/specializations/deep-learning) by [**DeepLearning.AI**](https://www.deeplearning.ai/) on **Coursera**, instructed by **Younes Bensouda Mourri** and **Łukasz Kaiser**. The specialization consists of 4 courses that covers state-of-the-art deep learning techniques needed to build cutting-edge NLP systems.

## 📋 Overview

This Natural Language Processing Specialization explores how algorithms process and analyze human language, a key area of machine learning used to extract insights from large, unstructured text and audio data. As AI grows, so does the need for experts in building NLP models.

By the end, you’ll be able to create applications for tasks like sentiment analysis, question-answering, translation, and summarization—essential skills for the AI-driven future.

The course is led by top experts: Younes Mourri (Stanford AI instructor) and Łukasz Kaiser (Google Brain researcher and co-author of the Transformer paper).

## 🎓 What You'll Learn

In this course, you’ll learn to:

- **Sentiment Analysis & Word Vectors**: Use logistic regression, naïve Bayes, and word vectors for sentiment analysis, completing analogies, word translation, and approximate nearest neighbors
  
- **Probabilistic Models**: Implement autocorrect, autocomplete, and part-of-speech tagging using dynamic programming, hidden Markov models, and word embeddings

- **Sequence Models**: Build deep learning models using recurrent neural networks, LSTMs, GRUs, and Siamese networks for sentiment analysis, text generation, and named entity recognition

- **Attention Models**: Implement machine translation, text summarization, and question-answering systems using encoder-decoder models, causal attention, and self-attention mechanisms

## 🛠️ Applied Learning Projects

Throughout this specialization, you'll work on practical projects that apply NLP techniques to real-world problems:

- Apply traditional models like logistic regression and naïve Bayes for tasks such as sentiment analysis and word translation
- Use dynamic programming and HMMs for autocorrect, autocomplete, and POS tagging
- Build deep learning models (RNNs, LSTMs, GRUs, Siamese networks) in TensorFlow for tasks like NER, text generation, and duplicate detection
- Implement advanced NLP with encoder-decoder architectures and attention mechanisms for translation, summarization, and question answering using models like T5 and BERT with Hugging Face Transformers

## 📋 Specialization Structure

The specialization consists of 5 courses:

### 📚 Course 1: Natural Language Processing with Classification and Vector Spaces

This course introduces foundational NLP concepts through classification techniques and vector space models. You'll learn to analyze sentiment, discover word relationships, and build a simple translation algorithm.

#### 📦 Modules:

**Module 1: Sentiment Analysis with Logistic Regression**
- Extract features from text into numerical vectors
- Build a binary classifier for tweets using logistic regression
- Implement preprocessing techniques for text data
- Evaluate model performance using accuracy metrics

**Module 2: Sentiment Analysis with Naïve Bayes**
- Learn the theory behind Bayes' rule for conditional probabilities
- Build a Naive Bayes tweet classifier from scratch
- Compare performance with logistic regression approaches
- Handle class imbalances in sentiment datasets

**Module 3: Vector Space Models**
- Create word vectors that capture semantic dependencies between words
- Apply dimensionality reduction with Principal Component Analysis (PCA)
- Visualize word relationships in two dimensions
- Discover semantic relationships between words in vector space

**Module 4: Machine Translation and Document Search**
- Transform word vectors for cross-language applications
- Implement locality sensitive hashing (LSH) for approximate nearest neighbor search
- Build a simple English to French translation algorithm using pre-computed word embeddings
- Create efficient document search systems using vector representations

#### 🔑 Key Projects:
- Twitter sentiment classifier using logistic regression
- Naive Bayes implementation for sentiment analysis
- Word relationship visualization using PCA
- English-French translation system using word embeddings and LSH

### 📚 Course 2: Natural Language Processing with Probabilistic Models

This course explores probabilistic approaches to NLP through autocorrect systems, part-of-speech tagging, language modeling, and word embeddings, providing a foundation in both classical and neural network-based NLP techniques.

#### 📦 Modules:

**Module 1: Autocorrect**
- Learn the principles of minimum edit distance and dynamic programming
- Build your own spellchecker for correcting misspelled words
- Implement efficient string matching algorithms
- Create data structures for fast word lookup and correction

**Module 2: Part of Speech Tagging and Hidden Markov Models**
- Understand Markov chains and their applications in NLP
- Build Hidden Markov Models (HMMs) for sequence labeling
- Implement the Viterbi algorithm for optimal sequence determination
- Create part-of-speech tagging systems for Wall Street Journal corpus

**Module 3: Autocomplete and Language Models**
- Develop N-gram language models for sequence probability calculation
- Build your own autocomplete system using Twitter corpus data
- Handle out-of-vocabulary words and sparse data challenges
- Implement smoothing techniques for better probability estimates

**Module 4: Word Embeddings with Neural Networks**
- Understand how word embeddings capture semantic meaning
- Build a Continuous Bag-of-Words (CBOW) model from scratch
- Create word embeddings from Shakespeare text
- Visualize and analyze semantic relationships in embedding space

#### 🔑 Key Projects:
- Custom spellchecker implementation using dynamic programming
- Part-of-speech tagger using Hidden Markov Models
- Twitter-based autocomplete system with N-gram language models
- Word2Vec implementation with Continuous Bag-of-Words architecture

### 📚 Course 3: Natural Language Processing with Sequence Models

This course explores advanced neural network architectures for NLP, focusing on sequential data processing through RNNs, LSTMs, GRUs, and Siamese networks to build sophisticated text generation and analysis systems.

#### 📦 Modules:

**Module 1: Recurrent Neural Networks for Language Modeling**
- Understand the limitations of traditional language models
- Learn how RNNs and GRUs process sequential data for text prediction
- Build a next-word generator using a simple RNN
- Train models on Shakespeare text data to generate synthetic text
- Implement techniques for handling long-range dependencies

**Module 2: LSTMs and Named Entity Recognition**
- Master Long Short-Term Memory (LSTM) networks
- Understand how LSTMs solve the vanishing gradient problem
- Build a Named Entity Recognition (NER) system using LSTMs
- Process and prepare Kaggle datasets for NER tasks
- Implement linear layers with LSTMs for improved performance
- Extract important information from text using NER techniques

**Module 3: Siamese Networks**
- Learn the architecture of Siamese networks with dual identical structures
- Understand similarity metrics and feature space mapping
- Build a Siamese network to identify duplicate questions
- Process and analyze the Quora Question Pairs dataset
- Implement contrastive loss functions for similarity learning
- Evaluate model performance on semantic similarity tasks

#### 🔑 Key Projects:
- Shakespeare text generator using RNNs and GRUs
- Named Entity Recognition system with LSTM architecture
- Question duplicate detector using Siamese networks
- Sentiment analysis with neural word embeddings

### 📚 Course 4: Natural Language Processing with Attention Models

This course covers advanced deep learning architectures for NLP, focusing on attention mechanisms and transformer models to build sophisticated systems for machine translation, text summarization, and question-answering.

#### 📦 Modules:

**Module 1: Neural Machine Translation**
- Understand the limitations of traditional sequence-to-sequence models
- Learn how attention mechanisms solve context and long dependency issues
- Build an English-to-Portuguese translator using encoder-decoder attention
- Implement and train attention models for improved translation quality
- Evaluate translation performance using BLEU scores

**Module 2: Text Summarization**
- Compare RNNs with the modern Transformer architecture
- Understand self-attention and multi-head attention mechanisms
- Build a Transformer model for text summarization
- Implement techniques for abstractive and extractive summarization
- Train and evaluate summarization models on news articles
- Handle long documents and maintain context in summaries

**Module 3: Question Answering**
- Explore transfer learning with state-of-the-art models
- Understand the architecture and capabilities of T5 and BERT
- Build question-answering systems using pre-trained models
- Fine-tune transformers for specific QA tasks
- Implement context-aware answer extraction techniques
- Evaluate QA systems using standard benchmarks

#### 🔑 Key Projects:
- English-Portuguese neural machine translator with attention
- News article summarizer using Transformer architecture
- Question-answering system with BERT and T5 models

## 🔧 Setup & Requirements

- 🐍 Python 3.x  
- 📐 NumPy
- 📊 Pandas
- 📈 Matplotlib
- 🔧 TensorFlow
- 🚀 Trax
- 📚 NLTK
- 🤖 Transformers
- 📦 Additional libraries as required by specific assignments

## 📖 How to Use This Repository

This repository serves as a resource for those taking the Natural Language Processing Specialization or for anyone interested in learning about NLP. You can use it to:

- 📂 Reference solutions to assignments  
- 🧠 Study code examples  
- 🛠️ Explore practical implementations of NLP concepts

## 🔗 Additional Resources

- [DeepLearning.AI](https://www.deeplearning.ai/)
- [Coursera NLP Specialization](https://www.coursera.org/specializations/natural-language-processing)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)

## 👨‍🏫 Instructors

- [**Andrew Ng**](https://www.linkedin.com/in/andrewyng/)  
  - 🧠 Founder of DeepLearning.AI  
  - 🌐 Co-founder of Coursera  
  - 🎓 Adjunct Professor at Stanford University

- [**Younes Bensouda Mourri**](https://www.linkedin.com/in/younes-bensouda-mourri-8749b9a9)  
  - 👨‍🏫 Instructor at DeepLearning.AI  
  - 🧠 Founder of LiveTech.AI  
  - 🎓 Lecturer at Stanford University

- [**Łukasz Kaiser**](https://www.linkedin.com/in/lukaszkaiser)  
  - 👨‍🏫 Instructor at DeepLearning.AI  
  - 🧪 Staff Research Scientist at Google Brain  

## ⚠️ Disclaimer

This repository is meant for educational purposes and as a reference. Please adhere to Coursera's Honor Code and avoid directly copying solutions for submission in the actual courses.

## 🙏 Acknowledgments

Special thanks to **DeepLearning.AI** and **Coursera** for creating such comprehensive learning materials.