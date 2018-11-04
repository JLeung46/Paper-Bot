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

The conversational model uses a RNN    

### Datasets

The intent model is trained on two separate datasets: one representing general human dialogues and another representing possible queries a user may make when requesting news. The dialogue data can be found in 'data/dialogue' and news data in 'data/news/newsData.csv'.

### Text Features



### Text Parsing
After user input is received, the intent model classifies the input as dialogue or news.
