# Paper-Bot
A conversational chat bot &amp; news article recommender.

#### Table of Contents
* [Overview](#overview)
* [Project Details](#details)
* [Installation](#installation)
* [Running](#running)
* [Results](#results)

## Overview

This project builds an intent based chatbot capable of handling general conversations as well as providing news article recommendations. The conversational component is built using a seq2seq RNN model from [DeepQA](https://github.com/Conchylicultor/DeepQA). News articles recommendations are provided using the [NewsAPI](https://newsapi.org/). The diagram below shows in more detail the entire pipeline.

![alt tag](pipeline.jpg)

## Project Details

### Modeling

The system contains two modeling components: an intent model and a conversational model. The intent model is responsible for classifying the given user input as dialogue or news. It is built using logistic regression. Logistic regression was chosen mainly due to its ease of use in addition to its surprisingly high accuracy (>97%) for this task. 

The conversational model uses an encoder-decoder architecture built using two LSTMs (commonly referred to as a seq2seq model). The encoder endcodes the input into a "thought vector" while the decoder decodes the output vector into a response. You can read more about seq2seq models here: 

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

### Datasets

The intent model is trained on two separate datasets: one representing general human dialogues and another representing possible queries a user may make when requesting news. The dialogue data can be found in `data/dialogue` and news data in `data/news/newsData.csv`. The newsData was generated manually, meaning I created the dataset by entering several queries that I thought someone would use when wanting to search for news articles. To enhance the size of the dataset, I left words unfilled in each of the queries which could be filled with possible keywords from categories coming from the NewsAPI. For example:

Unfilled Query: `Show me the latest new on <fill>.`

Then we can substitute `<fill>` with keywords such as `business`, `sports` or `CNN` which will give us a total of three queries:

`Show me the latest news on business.`

`Show me the latest news on sports.`

`Show me the latest news on CNN.`

The unfilled queries can be found in  `data/news/newsSamples.json`.

The seq2seq model is trained using the [Cornell Movie Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). The data is sorted into question answer pairs 

### Text Features



### Text Parsing
To extract keywords from the user's input I use the `nltk` library. I utilized nltk's named entity recognizer to extract locations and chunking to extract possible search words. One issue is that the nltk's NER system has trouble recognizing locations that aren't properly capitalized so if a user typed `Show me news related to business in the united states` then `united states` wouldn't be recognized since it wasn't properly cased. To solve this I used [Truecaser](https://github.com/nreimers/truecaser) to convert the input to its most probable casing.
