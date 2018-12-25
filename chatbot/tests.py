import os
import numpy as np
import torch
import torch.nn as nn

from text_data import *
from train_data import *
from seq2seq import *

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")


# Load vocabulary and query/response pairs
save_dir = os.path.join("data", "save")
text_data, pairs = load_data(corpus, corpus_name, datafile, save_dir)

# Print some pairs
print("\nSample Pairs:")
for pair in pairs[:10]:
    print(pair)
print("\n")

# Trim rare words
pairs = text_data.trim_rare_words(pairs,min_count=3) 

# Set size of hidden dim and vocabulary.
hidden_size = 500
vocab_size = 10000

# Get a batch of data
batch_size = 5
train_data = TrainData(text_data)
batches = train_data.batch_2_train_data([random.choice(pairs) for _ in range(batch_size)])
input_batch, lengths, target_variable, mask, max_target_len = batches

embedding = nn.Embedding(text_data.num_words, hidden_size)


def encoder_forward():
	"""
	Forward pass of encoder layer

	The input:
		input_batch: shape=(sentence_max_len, batch_size)
		lengths: shape=(batch_size)
		embeddings: shape=(len_vocab, hidden_size)

	The output: 
		output: shape=(sentence_max_len, batch_size, hidden_size)
	"""
	enc = EncoderRNN(embedding)
	output, hidden = enc.forward(input_batch, lengths)
	return output, hidden

def attn(attn_method='dot'):
	"""
	Attention layer

	The input to the attention layer are:
		enc_output: shape=(sentence_max_len, batch_size, hidden_size)
		decoder_hidden: shape=(1, batch_size, hidden_size)

	The output:
	 	attention_weights: shape=(batch_size, 1, sentence_max_len)
	"""
	attn_model = Attn(attn_method, hidden_size)

	# Set initial decoder hidden state as output of last hidden state of encoder
	enc_output, enc_hidden = encoder_forward()
	decoder_hidden = enc_hidden[:1]
	attn_weights = attn_model.forward(enc_output, decoder_hidden)
	return attn_weights

def decoder_luong_attention(attn_method='dot'):
	"""
	Luong attention layer
	
	The input to luong attention decoder:
		input_step: shape=(1, batch_size)
		last_hidden: shape=(n_layers*num_directions, batch_size, hidden_size)
		enc_output: shape=(sentence_max_len, batch_size, hidden_size)

	The output:
		decoder_output: shape=(batch_size, len_of_vocabulary)
		decoder_hidden: shape=(n_layers*num_directions, batch_size, hidden_size)


	"""
	attn_model = Attn(attn_method, hidden_size)
	# A single time step (one word) of input batch which is initialized with sos tokens
	input_step = torch.LongTensor([[text_data.sos_token for _ in range(batch_size)]])
	enc_output, enc_hidden = encoder_forward()
	last_hidden = enc_hidden[:1]
	decoder = LuongAttnDecoderRNN(attn_method, embedding, hidden_size, text_data.num_words)
	decoder_output, decoder_hidden = decoder.forward(input_step, last_hidden, enc_output)
	return decoder_output, decoder_hidden

if __name__ == '__main__':
	print("\n")
	print("Begin Tests...")
	print("\n")

	print(" Input Batch Size:", input_batch.size())
	print("Lengths:", lengths)

	# Check shape after encoder forward pass 
	assert list(encoder_forward()[0].size()) == [input_batch.size()[0], input_batch.size()[1], hidden_size]

	# Check shape after attention layer
	assert list(attn().size()) == [batch_size, 1, input_batch.size()[0]]

	# Check shape after luong attention decoder
	assert list(decoder_luong_attention()[0].size()) == [input_batch.size()[1], text_data.num_words]


	print('*********************')
	print('All Tests Passed!')


