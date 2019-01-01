import os

from chatbot import chatbot
from text_data import *

if __name__ == '__main__':
	corpus_name = "cornell movie-dialogs corpus"
	corpus_dir = os.path.join("data", corpus_name)
	datafile = os.path.join(corpus_dir, "formatted_movie_lines.txt")

	text_data = TextData()
	save_dir = os.path.join("data", "save")
	pairs = text_data.load_data()

	chatbot_trainer = chatbot.Chatbot(text_data=text_data)
	chatbot_trainer.main_train()