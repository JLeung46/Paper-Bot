# Paper-Bot
A conversational chat bot &amp; news article recommender.

#### Table of Contents
* [Overview](#overview)
* [Project Details](#project)
* [Installation](#installation)
* [Running](#running)
* [Results](#results)

## Overview

This project builds an intent based chatbot. The bot categorizes user input into two intents: dialogue or news. Depending on the category the bot will provide a response as if your were having a typical conversation with another human or it will provide recommendations for news articles. The motivation is to help improve the user experience by allowing the user to provide input naturally as if it were communicating to an acutal human being. Then by training the bot to understand natural human language, it would respond very similar as a human would. The project also provides a more personalized experience based on user preferences for news articles as articles are suggested based on the user's input. This avoids the need for having to scroll through hundreds of articles and reading any negative news that could put a damper on your day. The diagram below provides a high level overview of the system components and more details are explained below.

![alt tag](img/pipeline.jpg)

## Project Details

### Datasets

As shown in the above diagram, the system uses three different datasets (colored in gray). Two datasets are used train the intent model and a third datset is used to train the seq2seq model. More details on the modeling components are described later. The dialogue dataset contains sentences that occur in general human conversations whereas the news dataset contains sentences a person would say when requesting news articles.

Dialogue examples:

	Would you mind getting me a drink, Cameron?
	Where did he go?  He was just here.
	You might wanna think about it
	Did you change your hair?
	You know the deal.  I can't go if Kat doesn't go --
	Listen, I want to talk to you about the prom.


News examples:

	What's happening in Brazil?
	Show me the latest news happening in Brazil.
	Show me interesting news from Ynet.
	What top stories is the NY Times running right now?
	Can you show me news about business?
	Show me the newest entertainment related news.


A major challenge when building this system was finding enough data related to queries a person would enter when requesting news articles. To overcome this, a dataset was manually generated by entering several hypothetical queries that one would use when wanting to search for news articles. Additionally, to further enhance the size of the dataset, words were left unfilled in each of the queries which could later filled with possible keywords and categories common to news articles. Common news categories to fill the queries can be found on the [News Api](https://newsapi.org/).

For example:

	Unfilled Query: Show me the latest news on <fill>.

	Words such as business, sports or CNN can then be used to replace <fill> which will provide a total of three queries:

	Show me the latest news on business.

	Show me the latest news on sports.

	Show me the latest news on CNN.

The dialogue data is found in `data/dialogue.tsv` and news data in `data/news/newsData.csv`.

The unfilled queries are found in  `data/news/newsSamples.json`.

The code for generating for news dataset is found in `intent/news_data.py`

The third dataset used to train the seq2seq model was the [Cornell Movie Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).  The corpus contains 220,579 conversational exchanges between pairs of movie characters for a total of 304,713 utterances. The raw data is transformed into query/response pairs to be fed into the model. The raw data is found in `data/cornell` and formatted text after preprocessing is found in `data/formatted_movie_lines.txt`. More on the preprocessing steps are described below. 

### Preprocessing
To prepare the data to train an intent model, the dialogue and news datasets are combined into one dataset and are labled as `news` or `dialogue`. The two classes were balanced by matching the number of dialogue and news instances  as the amount of dialogue data available was substantually larger than the news data available. Steps to clean the raw text data include removing symbols using regex commands such as: `re.compile('[/(){}\[\]\|@,;]')`. Stopwords were also removed and a normalization step was taken by lowercasing all text. The code for cleaning and creating the intent dataset is found in `intent/intent_data.py`. 

To create a QA dataset to be fed into a seq2seq model using the Conell Movie Dialogs Corpus required a number of steps. The file `data/cornell/movie_conversations.txt` contains lines in the format `u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']`. Where `u0`and `u2` correspond to two character ID's and `m0` corresponds to the movie ID. The list following respresents line ID's that are said in sequence between the two characters. The actual lines that the line ID's map to are found in `data/cornell/movie_lines.txt`. As an example, using the above line we have:

	User0-L194: Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.
	User2-L195: Well, I thought we'd start with pronunciation, if that's okay with you.
	User0-L196: Not the hacking and gagging and spitting part.  Please.
	User2-L197: Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?

By using these sequences, a QA dataset is created which maps a characters query to another characters response to that query.  Credit for the preprocessing for Cornell Movie Dialog Corpus goes to [FloydHub]( https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus).

A common method to feed text data into a seq2seq model is building a vocabulary or dictionary. The dictionary maps each unique word found in the dataset to an index value. The index values can then be fed as input into the model and after the model produces an output, the indicies can then be converted back into words. Special tokens such as `PAD`, `SOS` and `EOS` are also added to the dictionary. The PAD token ensures sentences in a batch are the same length while the SOS and EOS tokens mark the start and end of a sentence. Additionally, the number of words in the dictionary could be massive so the size of the dictionary was limited by removing words below a certain frequency threshold and the max sentence length was set to ten to aid in training convergence. The code for preprocessing the data for the seq2seq model is found in `chatbot/text_data.py`.

### Text Features
Both the intent model and seq2seq model requires transforming the raw data into numerical values that the machine learning algorithm can understand.

To create features for the intent model, the dataset containing dialogue and news data was transformed into weight vectors using [tfidf](http://www.tfidf.com). Tfidf is a commonly used technique in text mining and natural language processing that assigns a weight to words to reflect how important a word is to a document. To keep it simple, tfidf works by assigning larger weights to words that occur frequently within a document and smaller weights to words that occur infreqently or frequently across documents. For example if the words 'cat' and 'dog' occur frequently in a document then those words would have a high weight since those words would be assigned to a document about animals. But there are also other words which occur frequently across documents that don't offer much meaning such as 'is', 'the', and 'then' and as a result are assigned smaller weights. All the words in the corpus are assigned weights and stacked together to form a weight vectors which can then be fed into the machine learning algorithm. 

The code for using tfidf is found in `intent/utils.py` and the resulting weights vectors are saved as a pickle file in `intent/tfidf_vectorizer.pkl`.

The seq2seq model learns a featurize representation of words, known as word embeddings. Word embeddings allow for NLP applications to understand semantic relationships and analogies between words such as man is to woman and king is to queen. In this system, the embeddings are learned using an embedding layer that assigns each word in the vocabulary a feature vector which represents properties of a word. These vectors can be used to measure semantic similarity between words. The word embeddings are learned through an embedding layer built within the seq2seq model. More details about the model is described below.


### Modeling
#### Logistic Regression
Logistic Regression was chosen for the intent model as it is fast, doesn't require many hyperparameters that need tuning and generally serves as a strong baseline model. Basically, the way logistic regression works is by first computing a dot product between features and their weights then uses the logit function to squash the output values between 0 and 1 to form probabilties. Training and testing the logistic regression classifier using a 90/10 split resulted in >97% accuracy.

The code for building the logistic regression model is found in `intent/intent_model.py` and the code for training it is found in `intentTrainer.py`. 

#### Seq2Seq

The conversational component of the chatbot uses an encoder-decoder architecture built using two recurrent neural networks. This is commonly referred to as a seq2seq model as it takes a variable length sequence as input and produces a variable length sequence as output. Seq2seq models have been used in a number of applications including machine translation, speech recognition and time series forecasting. 

The encoder-decoder architecture uses one RNN to act as an encoder which iterates through the input sequence one word at a time. For each time step (word in the input sequence), two outputs are produced: a output vector and a hidden state vector. The output vector is the prediction the model makes for the current time step whereas the hidden state captures useful information from both the current and previous time steps and carries it to other layers of the network. The interesting thing about RNNs is that they take information from previous hidden states to make better predictions at the current state by forming conditional probability distribution at each time step. For example, using the input sentence: 

	Joe and Sally went to the movie theater and ate <blank>.

A human would not read each word independently to predict the last word in the sentence but would use additional information from the context of previous words in order to make a prediction. This is exactly what a RNN does! Additionally, bi-directional RNNs not only read an input sequence from left to right but also from right to left because there are times when using information later in the sequence can help predict words earlier in the sentence. This is a very brief, high level explanation of RNNs but more details more can be found here:

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)

[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

After the encoder iterates through the entire input sequence, it produces a final output vector which "encodes" the input sequence. The encoded input is then fed into the decoder which is a separate RNN. This decoder network takes as input the encoded input sequence and is trained to produce the target response given the input sequence. So now the encoder learns a conditional probability distribution given the encoded input sequence. Say for example we have a query as input and a response as the target output:

	Input Query: Hi, how are you today?
	Target Response: I'm doing great and you?

The encoder is trained to learn features of the input query and outputs a encoded output vector. The encoded output vector is then fed into the decoder which is trained to learn a mapping between the encoded output vector and the target response. 

One issue that RRNs encounter is vanishing gradients due to very long sequences. Vanishing gradients is a problem that occurs with very deep networks which causes gradients to disappear when performing backprop because there are many multiplications being computed when trying to send gradients very deep in the network back to earlier layers in the network. Intuitively, the network may have a hard time making accurate predictions later in the network if the information needed to make an accurate prediction comes very early in the network. For example:
		
	The kids went to the movies and ate alot of popcorn and after riding the bumper cars were very sick.

Inorder to predict the word "were" the word "kids" needs to be known. So a network encountering vanishing gradients due to the long sentence would have a hard time memorizing the information from "kids" inorder to make the prediction "were".

One method to overcome vanishing gradients is to use Gated Recurrent Units (GRU). GRUs use a "memory cell" to memorize past information and passes this along the network to help make future predictions. Basically, a GRU has two gates: a reset and update gate. The reset gate determines how to combine the input and previous memory whereas the update gate controls how much of the previous memory to keep. Additionally, bi-directional GRUs are used in this system so the input sequence is read by the network both from left to right and from right to left.  More about GRUs can be learning here:

[Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
](https://arxiv.org/abs/1412.3555)

The code for building the seq2seq model is found in `chatbot/seq2seq.py`


### Information Extraction from Text
Inorder to send requests to the News API, relevant information needs to be extracted from the raw user input. Additionally, certain rules must be followed according to the News API such as not being able to mix the sources parameter with the country or category parameter. This can be solved with simple if else statements and to get the list of available news categories, countries and publishers supported by the news api, a request was sent for each of the parameters and saved in `data/news`. Then after a user enters a query, the query can be tokenized and a simple search for any of those news categories, countries or publishers can be performed.

A more interesting problem involved extracting potential search keywords not specified from the News API. For example, using the folllowing queries:

	Show me the latest news related to the iphone.
	Show me the latest news related to basketball.
    What is the latest on Donald Trump?
    What's new about bitcoin?
    What's the news on Kim Kardashian?

The goal then is to extract the keywords `iphone`, `basketball`, `Donald Trump`, `bitcoin`, `Kim Kardashian`. This is a typical entity extraction task which involves searching for potentially interesting entities in a sentence. The approach taken in this system is called chunking and works well since the news dataset is relatively small and the queries aren't overly complex. Chunking works by first assigning part of speech tags to each word in the sentence. Next, a chunk grammar is defined to indicate how sentences should be chunked. A chunk grammar can be defined using simple regular expression rules. For example, when trying to extract the keys words from the above sentences the procedure would look at follows: first apply POS tagging:

	[('show', 'VB'), ('me', 'PRP'), ('the', 'DT'), ('latest', 'JJS'), ('news', 'NN'), ('related', 'VBN'), ('to', 'TO'), ('the', 'DT'), ('iphone', 'NN'), ('.', '.')]

	[('What', 'WP'), ("'s", 'VBZ'), ('the', 'DT'), ('news', 'NN'), ('on', 'IN'), ('Kim', 'NNP'), ('Kardashian', 'NNP'), ('?', '.')]

Next, define chunk grammar rules:

	<DT><NN>
	<NNP>+

The first grammar rule says to find chunks with a determiner <DT> follow by a noun <NN>.

The second grammar rule says to find chunks with one or more proper nouns <NNP>.

The two rules can then extract the two search keywords "iphone" and "Kim Kardashian".

The code for extracting keywords from the raw user input is found in `app/text_parser.py`
More about information extraction and chunking can be learned from the [nltk book](https://www.nltk.org/book/ch07.html).

A side note: an issue encountered using nltk's NER system when attempting to extract names of countries was it had trouble recognizing countries that aren't properly capitalized. For example, if a user typed `Show me news on the iphone in the united states` then `united states` wouldn't be recognized since it wasn't properly cased. As a work around, [Truecaser](https://github.com/nreimers/truecaser) was used as a preprocessing step to convert the input to its most probable casing before applying NER.


### Installation
The program requires the following dependencies:

 * python 3
 * PyTorch
 * numpy
 * pandas
 * sklearn
 * CUDA
 * nltk
 * tqdm
 * flask
 * requests
 * iso3166

After installing Python 3 and PyTorch, run `pip3 install -r requirements.txt` to install all other dependencies.


### Running
Before running the app we need to download some files:

For truecaser we need the pre-trained `distributions.obj` which contains the frequencies of unigrams, bigrams and trigrams which can be downloaded [here](https://github.com/nreimers/truecaser/releases) and placed in the `truecaser` directory.

The pre-trained seq2seq model can be downloaded [here](https://drive.google.com/open?id=1Iqp-ZnT2DVslKw9bIPgMbTUodBeCIkqc) and placed in the `save` directory.

To run the app:
```
cd app
python webapp.py
```
The app should then be running on [http://localhost:8080/](http://localhost:8080/).

![alt tag](img/app.JPG)

The chatbot can also be run in a jupyter notebook in `notebooks/chatbot_notebook.py`.

### Results

Here's a sample dialog when running the app:

> **me:** Hi  
> **bot:** Hi.
> 
> **me:** How are you?  
> **bot:** Okay.
> 
> **me:** What's your name?  
> **bot:** Valdes. miss valdes.
> 
> **me:** show me the latest news in sports.  
> **bot:** 
> 
> Title: 
> Warriors takeaways: What we learned in 117-101 win over Grizzlies
> 
> Description: 
> Alfonzo McKinnie was exceptional once again, but the Warriors still had some issues Monday night against the Grizzlies.
> 
> URL: 
> https://www.nbcsports.com/bayarea/warriors/warriors-takeaways-what-we-learned-117-101-win-over-grizzlies
> 
> --------------------
> 
> Title: 
> Jimmy Butler turns away questions about LA future after loss to Clips
> 
> Description: 
> Jimmy Butler's present and potential future landing spot collided at Staples Center, where the Clippers beat Butler and the Timberwolves 120-109. But Butler wasn't interested in
> expounding on his potential future in L.A.
> 
> URL: 
> http://www.espn.com/nba/story/_/id/25191907/jimmy-butler-minnesota-timberwolves-turns-away-questions-future-los-angeles
> 
> --------------------
> 
> **me:**do you know where my cat went?  
> **bot:** I do n't know.
> 
>**me:**what news is there on the iphone?  
> **bot:** 
> 
> Title: 
> Apple's market cap fell below $1 trillion and its iPhone XR woes just clobbered stocks of its China suppliers too
> 
> Description: 
> Shares of Apple's suppliers across Asia were...
> 
> URL: 
> http://uk.businessinsider.com/apple-drops-below-1-trillion-clobbering-shares-of-china-suppliers-2018-11
> 
> --------------------
> 
> Title: 
> Apple's market cap fell below $1 trillion and its iPhone XR woes just clobbered stocks of its China suppliers too
> 
> Description: 
> Apple’s Asian suppliers fell after a report that Apple will halt additional production lines for the iPhone XR. The news sent Apple below $1 trillion.
> 
> URL: 
> https://www.businessinsider.com/apple-drops-below-1-trillion-clobbering-shares-of-china-suppliers-2018-11
> 
> --------------------
> 

### Train your own model
To train your own seq2seq model, simply run `seq2seq_trainer.py`. Model parameters may be adjusted in `chatbot/chatbot.py`.
