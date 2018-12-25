"""
Builds vocabulary dictionaries and performs pre-processing.
"""
import os
import unicodedata
import re
from app import settings


BASE_DIR = settings.BASE_DIR
CORPUS_NAME = "cornell" # Name of corpus
CORPUS_DIR = os.path.join("data", CORPUS_NAME) # Directory of corpus
DATA_DIR = os.path.join(BASE_DIR, CORPUS_DIR)
DATAFILE = os.path.join(DATA_DIR, "formatted_movie_lines.txt") # Directory for formatted movie lines


class TextData:
	"""
	Class for the dataset.
	"""
	def __init__(self):
		"""
		Builds the vocabulary dictionaries
		"""
		self.corpus_name = CORPUS_NAME # Name of corpus
		self.corpus_dir = CORPUS_DIR # Directory of corpus

		self.pad_token = 0 # padding
		self.sos_token = 1 # start of sentence
		self.eos_token = 2 # end of sentence

		self.trim = False
		self.word2idx = {}
		self.word2count = {}
		self.idx2word = {self.pad_token: "PAD", self.sos_token: "SOS", self.eos_token: "EOS"}
		self.num_words = 3 # count the pad, sos, eos tokens

		self.max_length = 10 # Max sentence length

		self.pairs = None # query/response pairs

	def add_sentence(self, sentence):
		"""
		Calls add_word for each word in a sentence.

		Parameters:
			sentence (String): A string of text.
		"""
		for word in sentence.split(' '):
			self.add_word(word)

	def add_word(self, word):
		"""
		Adds words to the vocabulary and 
		builds the dictionaries.

		Parameters:
			word (String): A word.
		"""
		if word not in self.word2idx:
			self.word2idx[word] = self.num_words
			self.word2count[word] = 1
			self.idx2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2count[word] += 1

	def trim_words(self, min_count):
		"""
		Removes words below a certain threshold.

		Parameters:
			min_count (int): The threshold number.
		"""
		if self.trim:
			return
		self.trim = True

		keep_words = []

		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		print('keep_words {} / {} = {:.4f}'.format(
			len(keep_words), len(self.word2idx), len(keep_words) / len(self.word2idx)))

		self.word2idx = {}
		self.word2count = {}
		self.idx2word = {self.pad_token: "PAD", self.sos_token: "SOS", self.eos_token: "EOS"}
		self.num_words = 3

		# Rebuild the dictionaries using newly thresholded words.
		for word in keep_words:
			self.add_word(word)

	def trim_rare_words(self, pairs, min_count=3):
		"""
		Remove rarely used words out of vocabulary.

		Parameters:
			pairs (list): A list of query/response pairs.
			min_count (int): The threshold number.
		
		Returns:
			keep_pairs (list): A list of trimmed query/response pairs.
		"""
		self.trim_words(min_count)
		keep_pairs = []
		for pair in pairs:
			input_sentence = pair[0]
			output_sentence = pair[1]
			keep_input = True
			keep_output = True

			for word in input_sentence.split(' '):
				if word not in self.word2idx:
					keep_input = False
					break
			for word in output_sentence.split(' '):
				if word not in self.word2idx:
					keep_output = False
					break

			if keep_input and keep_output:
				keep_pairs.append(pair)

		print("Trimmed from {} pairs to {}, {:.4f} of total"
			.format(len(pairs), len(keep_pairs),
			 len(keep_pairs) / len(pairs)))

		return keep_pairs


	def load_data(self):
		"""
		Load and prepare the data. 

		Parameters:
			save_dir (String): Directory where to save the prepared data.

		Returns:
			text_data (TextData) : TextData object
			pairs (list): List of query/response pairs.
		"""
		print("Preparing training data...")
		pairs = read_corpus(DATAFILE)
		print("Read {!s} sentence pairs".format(len(pairs)))
		pairs = filter_pairs(pairs)
		print("Trimmed to {!s} sentence pairs".format(len(pairs)))
		print("Counting words...")
		print("Counted words:", self.num_words)
		for pair in pairs:
			self.add_sentence(pair[0])
			self.add_sentence(pair[1])
		self.pairs = self.trim_rare_words(pairs)
		return pairs

def unicode_to_ascii(text):
	"""
	Converts a unicode string to ASCII.

	Parameters:
		text (String): A unicode string.
	Returns:
		An ASCII version of the string.
	"""
	return ''.join(
		char for char in unicodedata.normalize('NFD', text)
		if unicodedata.category(char) != 'Mn'
		)

def normalize_string(text):
	"""
	Lowercase, trim and remove non-letter characters.

	Parameters:
		text (String): A String of text.
	Returns:
		text (String): The normalized string.
	"""
	text = unicode_to_ascii(text.lower().strip())
	text = re.sub(r"([.!?])", r" \1", text)
	text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
	text = re.sub(r"\s+", r" ", text).strip()
	return text

def read_corpus(file_dir):
	"""
	Read query/response pairs.
		Parameters:
			file_dir (String): Directory of query/response pairs data.

		Returns:
			text_data (TextData): TextData object
	"""
	print("Reading lines..")
	lines = open(file_dir, encoding='utf-8').\
		read().strip().split('\n')
	pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]
	return pairs

def filter_pair(pair, max_length=10):
	"""
	Check if sentences in a pair are below the max_length threshold.

	Parameters: 
		pair (list): A pair of sentences.
		max_length (int): Threshold value for max length of sentence.

	Return:
		Boolean: Whether pair is under threshold.
	"""
	return len(pair[0].split(' ')) < max_length and len(pair[1].split(' ')) < max_length

def filter_pairs(pairs):
	"""
	Calls filter_pair on all pairs.

	Parameters:
		pairs (list): A list of sentence pairs.

	Returns:
		list: A list of filtered sentence pairs.
	"""
	return [pair for pair in pairs if filter_pair(pair, )]
















