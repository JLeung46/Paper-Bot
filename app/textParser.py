import sys
import os
import pickle
import json
import requests
from iso3166 import countries

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('maxent_ne_chunker')
nltk.download('words')

import settings

TRUECASER_DIR = os.path.join(settings.BASE_DIR, 'truecaser')

sys.path.append(os.path.join(settings.BASE_DIR, 'intent'))
sys.path.append(TRUECASER_DIR)

from newsData import NewsData
from truecaser import Truecaser

GRAMMAR = ('''
        search_keyword: {<DT>?<IN><NN>*}
        {<DT><JJ>?<NN>*}
        ''')


class TextParser:

    def __init__(self):

        self.newsData = NewsData(os.path.join(settings.BASE_DIR, 'data'))
        self.sourceDict = self.loadSourceDict(os.path.join(
            settings.BASE_DIR, 'data' + os.sep + 'news'
            + os.sep + 'newsSources.json'))

        self.truecaser = open(TRUECASER_DIR
                              + os.sep + 'distributions.obj', 'rb')
        self.uniDist = pickle.load(self.truecaser)
        self.backwardBiDist = pickle.load(self.truecaser)
        self.forwardBiDist = pickle.load(self.truecaser)
        self.trigramDist = pickle.load(self.truecaser)
        self.wordCasingLookup = pickle.load(self.truecaser)

        self.defaultUrl = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=b806b49b7ae045cf8db591b6c8b7d4b5'
        self.baseUrl = 'https://newsapi.org/v2/top-headlines?'
        self.apiUrl = 'apiKey=b806b49b7ae045cf8db591b6c8b7d4b5'

        self.keywordUrl ='q='
        self.countryUrl = 'country='
        self.sourceUrl = 'sources='
        self.categoryUrl = 'category='

        self.grammar = None
        self.keyword = ''
        self.category = ''
        self.country = ''
        self.source = ''

    def setGrammar(self, grammar):
        self.grammar = grammar

    def getCountry(self, word_tokens):
        true_case_tokens = Truecaser.getTrueCase(word_tokens, 'title', self.wordCasingLookup, self.uniDist, self.backwardBiDist, self.forwardBiDist, self.trigramDist)
        for chunk in nltk.ne_chunk(nltk.pos_tag(true_case_tokens)):
            if hasattr(chunk, 'label'):
                if chunk.label() == 'GPE':
                    country = ' '.join(c[0] for c in chunk)
                    try:
                        self.country = self.getCountryId(country)
                    except KeyError:
                        self.country = ""

    def getKeyword(self, word_tokens):
        self.setGrammar(GRAMMAR)
        chunkParser = nltk.RegexpParser(self.grammar)
        pos = nltk.pos_tag(word_tokens)
        tree = chunkParser.parse(pos)
        for subtree in tree.subtrees(filter=lambda x: x.label() == 'search_keyword'):
            for leaf in subtree.leaves():
                if leaf[1] =='NN':
                    self.keyword = leaf[0]

    def getSource(self, word_tokens):
        for word in word_tokens:
            if word in self.newsData.publishers:
                source = word
                self.source = self.getSourceId(source, self.sourceDict)

    def getCategory(self, word_tokens):
        for word in word_tokens:
            if word in self.newsData.categories:
                self.category = word

    def loadSourceDict(self, filename):
        with open(filename) as f:
            sourceDict = json.load(f)
        return sourceDict

    def getSourceId(self, source_name, source_dict):
        sourceId = source_dict[source_name]['id']
        return sourceId

    def getCountryId(self, country_name):
        return countries.get(country_name).alpha2.lower()

    def parseText(self, sentence):

        word_tokens = nltk.word_tokenize(sentence)
        self.getKeyword(word_tokens)
        self.getCountry(word_tokens)
        self.getSource(word_tokens)
        self.getCategory(word_tokens)

    def generateUrl(self, sentence):

        completeUrl = [self.baseUrl]

        self.parseText(sentence)

        if self.keyword:
            completeUrl.append(self.keywordUrl + self.keyword + "&")
            completeUrl.append(self.apiUrl)
            return ''.join(completeUrl)


        if self.country:
            completeUrl.append(self.countryUrl + self.country + "&")
            if self.category:
                completeUrl.append(self.categoryUrl + self.category + "&")

        elif self.category:
            completeUrl.append(self.categoryUrl + self.category + "&")

        elif self.source:
            completeUrl.append(self.sourceUrl + self.source + "&")

        completeUrl.append(self.apiUrl)

        return ''.join(completeUrl)

    def filterResponse(self, response, num_articles=5):
        articles = []
        for idx, article in enumerate(response['articles']):
            if idx == num_articles:
                break
            articles.append(dict(title=article['title'], description=article['description'], url=article['url']))
        return articles

    def displayResponse(self, response):
        articles = self.filterResponse(response)
        strResponse = ''
        for article in articles:
            if not article['description']:
                article['description'] = 'None'
            # To Do: Add html tags using JS file instead of here
            strResponse += '<u>Title</u>: ' + '<br/>' + article['title'] + '<br/><br/>' + '<u>Description</u>: ' + '<br/>' + article['description'] + '<br/><br/>' + '<u>URL</u>: ' + '<br/> <a href="' + article['url'] + '">' + article['url'] +'</a>' + '<br/><br/>' + '--------------------' + '<br/><br/>'
        return strResponse

    def getResponse(self, sentence):
        url = self.generateUrl(sentence)
        print(url)
        response = requests.get(url).json()
        print(response)
        response = self.displayResponse(response)
        return response
