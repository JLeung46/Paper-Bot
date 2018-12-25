import os

from chatbot import *
from text_data import *

if __name__ == '__main__':
	corpus_name = "cornell movie-dialogs corpus"
	corpus_dir = os.path.join("data", corpus_name)
	datafile = os.path.join(corpus_dir, "formatted_movie_lines.txt")

	text_data = TextData(corpus_name, corpus_dir)
	save_dir = os.path.join("data", "save")
	pairs = text_data.load_data(datafile, save_dir)

	chatbot = Chatbot(text_data=text_data)
	chatbot.main_train()