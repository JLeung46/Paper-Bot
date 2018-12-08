"""
This class extracts tokens from the user input classified as "news"
The tokens extracted are based off of news categories, publisher names and
countries supported from the newsAPI: https://newsapi.org/
"""

import sys
import os
import pickle
import json
import requests
from iso3166 import countries

import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')

import settings

TRUECASER_DIR = os.path.join(settings.BASE_DIR, 'truecaser')

sys.path.append(os.path.join(settings.BASE_DIR, 'intent'))
sys.path.append(TRUECASER_DIR)

from news_data import NewsData
from truecaser import Truecaser

# Grammar rule for search keywords
GRAMMAR = ('''
        search_keyword: {<DT>?<IN><NN>*}
        {<DT><JJ>?<NN>*}
        ''')


class TextParser:
    """
    This class extracts constructs a URL from tokens from user input and
    sends a request to the newsAPI.
    """

    def __init__(self):
        """
        The constructor for the TextParser class which constructs a URL to be sent
        to the newsAPI.
        """

        self.news_data = NewsData(os.path.join(settings.BASE_DIR, 'data'))
        self.source_dict = load_source_dict(os.path.join(
            settings.BASE_DIR, 'data' + os.sep + 'news'
            + os.sep + 'newsSources.json'))

        self.truecaser = open(TRUECASER_DIR
                              + os.sep + 'distributions.obj', 'rb')
        self.uni_dist = pickle.load(self.truecaser)
        self.backward_bi_dist = pickle.load(self.truecaser)
        self.forward_bi_dist = pickle.load(self.truecaser)
        self.trigram_dist = pickle.load(self.truecaser)
        self.word_casing_lookup = pickle.load(self.truecaser)

        self.default_url = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=\
                            b806b49b7ae045cf8db591b6c8b7d4b5'
        self.base_url = 'https://newsapi.org/v2/top-headlines?'
        self.api_url = 'apiKey=b806b49b7ae045cf8db591b6c8b7d4b5'

        self.keyword_url = 'q='
        self.country_url = 'country='
        self.source_url = 'sources='
        self.category_url = 'category='

        self.grammar = None
        self.keyword = ''
        self.category = ''
        self.country = ''
        self.source = ''

    def set_grammar(self, grammar):
        """
        Defines the grammar rules to extract potential search keywords.

        Parameters:
            grammar (String): A grammar structure

        """
        self.grammar = grammar

    def get_country(self, word_tokens):
        """
        Uses Truecaser to properly case word tokens then extracts the countries.

        Parameters:
            word_tokens (list): A list of word tokens.
        """
        true_case_tokens = Truecaser.getTrueCase(word_tokens, 'title', self.word_casing_lookup,
                                                 self.uni_dist, self.backward_bi_dist,
                                                 self.forward_bi_dist, self.trigram_dist)
        for chunk in nltk.ne_chunk(nltk.pos_tag(true_case_tokens)):
            if hasattr(chunk, 'label'):
                if chunk.label() == 'GPE':
                    country = ' '.join(c[0] for c in chunk)
                    try:
                        self.country = get_country_id(country)
                    except KeyError:
                        self.country = ""

    def get_keyword(self, word_tokens):
        """
        Extracts potential search keywords using grammar rules.

        Parameters:
            word_tokens (list): A list of word tokens.
        """
        self.set_grammar(GRAMMAR)
        chunkParser = nltk.RegexpParser(self.grammar)
        pos = nltk.pos_tag(word_tokens)
        tree = chunkParser.parse(pos)
        for subtree in tree.subtrees(filter=lambda x: x.label() == 'search_keyword'):
            for leaf in subtree.leaves():
                if leaf[1] == 'NN':
                    self.keyword = leaf[0]

    def get_source(self, word_tokens):
        """
        Extracts news publisher names.

        Parameters:
            word_tokens (list): A list of word tokens.

        """
        for word in word_tokens:
            if word in self.news_data.publishers:
                source = word
                self.source = get_source_id(source, self.source_dict)

    def get_category(self, word_tokens):
        """
        Extracts news categories.

        Parameters:
            word_tokens (list): A list of word tokens.
        """
        for word in word_tokens:
            if word in self.news_data.categories:
                self.category = word

    def parse_text(self, sentence):
        """
        Extracts tokens from a sentence.

        Parameters:
            sentence (String): A string of text.
        """

        word_tokens = nltk.word_tokenize(sentence)
        self.get_keyword(word_tokens)
        self.get_country(word_tokens)
        self.get_source(word_tokens)
        self.get_category(word_tokens)

    def generate_url(self, sentence):
        """
        Creates an URL to be sent to the newsAPI.

        Parameters:
            sentence (String): A string of text.

        Returns:
            complete_url (String): A string representing a URL.
        """

        complete_url = [self.base_url]

        self.parse_text(sentence)

        if self.keyword:
            complete_url.append(self.keyword_url + self.keyword + "&")
            complete_url.append(self.api_url)
            return ''.join(complete_url)


        if self.country:
            complete_url.append(self.country_url + self.country + "&")
            if self.category:
                complete_url.append(self.category_url + self.category + "&")

        elif self.category:
            complete_url.append(self.category_url + self.category + "&")

        elif self.source:
            complete_url.append(self.source_url + self.source + "&")

        complete_url.append(self.api_url)

        return ''.join(complete_url)

    def get_response(self, sentence):
        """
        Takes in raw user input and displays a list of related news articles.

        Parameters:
            sentence (String): The raw user input.

        Returns:
            response (String): A string of  news articles related to user's input.
        """
        url = self.generate_url(sentence)
        print(url)
        response = requests.get(url).json()
        print(response)
        response = display_response(response)
        return response

def load_source_dict(filename):
    """
    Loads a dictionary which maps publisher names to their id.

    Parameters:
        filename (String): Directory of where the source dict is located.
    """
    with open(filename) as file:
        source_dict = json.load(file)
    return source_dict

def get_source_id(source_name, source_dict):
    """
    Returns the id for a publisher name.

    Parameters:
        source_name (String): The name of the publisher.
        source_dict (dict): The dictionary which maps publisher names to ids.

    Returns:
        source_id (String): The id for the publisher
    """
    source_id = source_dict[source_name]['id']
    return source_id

def get_country_id(country_name):
    """
    Returns the ISO 3166 code for a country.

    Parameters:
        country_name (String): The name of the country.

    Returns:
        country_id (String): The ISO 3166 code for the country.
    """
    country_id = countries.get(country_name).alpha2.lower()
    return country_id

def filter_response(response, num_articles=5):
    """
    Specifies the number of news articles and information returned.

    Parameters:
        response: (dictionary): A dictionary containing articles and their meta data.
        num_articles (int): Number of articles to return.
    Returns:
        articles (list): A list of articles.
    """
    articles = []
    for idx, article in enumerate(response['articles']):
        if idx == num_articles:
            break
        articles.append(dict(title=article['title'], description=article['description'],
                             url=article['url']))
    return articles

def display_response(response):
    """
    Specifies the format in which the articles are displayed to the user.

    Paramaters:
        reponse (json): A json object returned from the newsAPI.

    Returns:
        str_response (String): A formatted string that is displayed to the user.
    """
    articles = filter_response(response)
    str_response = ''
    for article in articles:
        if not article['description']:
            article['description'] = 'None'
        # To Do: Add html tags using JS instead of here
        str_response += '<u>Title</u>: ' + '<br/>' + article['title'] + '<br/><br/>' +\
                        '<u>Description</u>: ' + '<br/>' + article['description'] +\
                        '<br/><br/>' + '<u>URL</u>: ' + '<br/> <a href="' + article['url'] +\
                        '">' + article['url'] +'</a>' + '<br/><br/>' + '--------------------' +\
                        '<br/><br/>'
    return str_response
