"""
Initialize encoder and decoder models. Choose to train
from scratch or load from checkpoint.
"""
import os
import sys
import torch.nn as nn
from torch import optim

from app import settings
BASE_DIR = settings.BASE_DIR

sys.path.append(os.path.join(BASE_DIR, "chatbot"))

from seq2seq import *
from train_data import *
from text_data import *
from greedy_search import *
from loss import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Chatbot:
	def __init__(self, text_data, model_name='chatbot_model', attn_method='dot', mode='train', hidden_size=500, encoder_num_layers=2, decoder_num_layers=2, 
		dropout=0.1, batch_size=64, clip=50.0, teacher_forcing_ratio=1.0, learning_rate=0.0001, decoder_learning_ratio=5.0, 
		n_iteration=8000, load_filename=None, checkpoint_iter=8000, save_dir=None, save_every=500, print_every=1):

		self.train_data = TrainData(text_data)
		self.text_data = self.train_data.text_data

		# Configure models
		self.model_name = model_name
		self.corpus_name = "cornell movie-dialogs corpus"
		self.attn_method = attn_method

		self.hidden_size = hidden_size
		self.encoder_num_layers = encoder_num_layers
		self.decoder_num_layers = decoder_num_layers
		self.dropout = dropout
		self.batch_size = batch_size

		self.clip = clip
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.learning_rate = learning_rate
		self.decoder_learning_ratio = decoder_learning_ratio
		self.n_iteration = n_iteration

		self.encoder = None
		self.decoder = None
		self.embedding = nn.Embedding(self.text_data.num_words, self.hidden_size)

		self.encoder_optimizer = None
		self.decoder_optimizer = None

		# Set checkpoint to load from; set to None if starting from scratch
		self.load_filename = load_filename
		self.checkpoint = None
		self.checkpoint_iter = checkpoint_iter

		# Checkpoint state dicts
		self.encoder_optimizer_sd = None
		self.decoder_optimizer_sd = None

		self.save_dir = "save"
		self.save_every = save_every
		self.print_every = print_every

		self.load_filename = os.path.join(BASE_DIR, self.save_dir, self.model_name, self.corpus_name,
	                            '{}-{}_{}'.format(self.encoder_num_layers, self.decoder_num_layers, self.hidden_size),
	                            '{}_checkpoint.tar'.format(self.checkpoint_iter))

	def load_model(self):
		# Load model if a loadFilename is provided
		if self.load_filename:
		    # If loading on same machine the model was trained on
		    self.checkpoint = torch.load(self.load_filename)
		    # If loading a model trained on GPU to CPU
		    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
		    encoder_sd = self.checkpoint['en']
		    decoder_sd = self.checkpoint['de']
		    self.encoder_optimizer_sd = self.checkpoint['en_opt']
		    self.decoder_optimizer_sd = self.checkpoint['de_opt']
		    embedding_sd = self.checkpoint['embedding']
		    self.text_data.__dict__ = self.checkpoint['voc_dict']

		print('Building encoder and decoder ...')
		# Initialize word embeddings
		if self.load_filename:
		    self.embedding.load_state_dict(embedding_sd)
		# Initialize encoder & decoder models
		self.encoder = EncoderRNN(self.embedding, self.hidden_size, self.encoder_num_layers, self.dropout)
		self.decoder = LuongAttnDecoderRNN(self.attn_method, self.embedding, self.hidden_size, self.text_data.num_words, self.decoder_num_layers, self.dropout)
		if self.load_filename:
		    self.encoder.load_state_dict(encoder_sd)
		    self.decoder.load_state_dict(decoder_sd)
		# Use appropriate device
		self.encoder = self.encoder.to(device)
		self.decoder = self.decoder.to(device)

		print('Building optimizers ...')
		self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
		self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate * self.decoder_learning_ratio)
		if self.load_filename:
			self.encoder_optimizer.load_state_dict(self.encoder_optimizer_sd)
			self.decoder_optimizer.load_state_dict(self.decoder_optimizer_sd)
		print('Models built and ready to go!')

	def train(self, input_variable, lengths, target_variable, mask, max_target_len):

		# Zero gradients
		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()

		# Set device options
		input_variable = input_variable.to(device)
		lengths = lengths.to(device)
		target_variable = target_variable.to(device)
		mask = mask.to(device)

		# Initialize variables
		loss = 0
		print_losses = []
		n_totals = 0

		# Forward pass encoder
		encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

		# Create initial decoder input
		decoder_input = torch.LongTensor([[self.text_data.sos_token for _ in range(self.batch_size)]])
		decoder_input = decoder_input.to(device)

		# Set initial decoder hidden state to the encoder's final hidden state
		decoder_hidden = encoder_hidden[:self.decoder.n_layers]

		# Teacher forcing
		use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

		# Forward batch of sequences through decoder one time step at a time
		if use_teacher_forcing:
			for t in range(max_target_len):
				decoder_output, decoder_hidden = self.decoder(
					decoder_input, decoder_hidden, encoder_outputs
					)
				# Teacher forcing: next input is current target
				decoder_input = target_variable[t].view(1, -1)
				# Calculate and accumulate loss
				mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
				loss += mask_loss
				print_losses.append(mask_loss.item() * nTotal)
				n_totals += nTotal
		else:
			for t in range(max_target_len):
				decoder_output, decoder_hidden = self.decoder(
					decoder_input, decoder_hidden, encoder_outputs
				)
				# No teacher forcing: next input is decoder's own current output
				_, topi = decoder_output.topk(1)
				decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
				decoder_input = decoder_input.to(device)
				# Calculate and accumulate loss
				mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
				loss += mask_loss
				print_losses.append(mask_loss.item() * nTotal)
				n_totals += nTotal

		# Perform backprop
		loss.backward()

		# Clip gradients
		_ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
		_ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

		# Adjust model weights
		self.encoder_optimizer.step()
		self.decoder_optimizer.step()

		return sum(print_losses) / n_totals

	def trainIters(self):
		"""
		Run training for n iteratons.
		"""

		# Load batches for each iteration
		training_batches = [self.train_data.batch_2_train_data([random.choice(self.text_data.pairs) for _ in range(self.batch_size)])
		for _ in range(self.n_iteration)]

		# Initializations
		print('Initializing..')
		start_iteration = 1
		print_loss = 0
		if self.load_filename:
			start_iteration = self.checkpoint['iteration'] + 1

		# Training Loop 
		print('Training..')
		for iteration in range(start_iteration, self.n_iteration + 1):
			training_batch = training_batches[iteration -1]
			# Extract field from batch
			input_variable, lengths, target_variable, mask, max_target_len = training_batch

			# Run training iteration with batch
			loss = self.train(input_variable, lengths, target_variable, mask, max_target_len)

			print_loss += loss

			# Print progress
			if iteration % self.print_every == 0:
				print_loss_avg = print_loss / self.print_every
				print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}"
					.format(iteration, iteration / self.n_iteration * 100, print_loss_avg))
				print_loss = 0

			# Save checkpoint
			if (iteration % self.save_every == 0):
				directory = os.path.join(self.save_dir, self.model_name, self.text_data.corpus_name, '{}-{}_{}'
					.format(self.encoder_num_layers, self.decoder_num_layers, self.hidden_size))
				if not os.path.exists(directory):
					os.makedirs(directory)
				torch.save({
					'iteration': iteration,
					'en': self.encoder.state_dict(),
					'de': self.decoder.state_dict(),
					'en_opt': self.encoder_optimizer.state_dict(),
					'de_opt': self.decoder_optimizer.state_dict(),
					'loss': loss,
					'voc_dict': self.text_data.__dict__,
					'embedding': self.embedding.state_dict()
					}, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

	def main_train(self):
		# Load model parameters
		self.load_model()

		# Ensure dropout layers are in train mode
		self.encoder.train()
		self.decoder.train()

		# Run training iterations
		print("Starting Training!")
		self.trainIters()

	def evaluate(self, sentence):
		# Set encoder and decoder to eval mode
		self.encoder.eval()
		self.decoder.eval()
		# Normalize input sentence
		sentence = normalize_string(sentence)
		# Convert words to indexes
		idx_batch = [self.train_data.idx_from_sentence(sentence)]
		lengths = torch.tensor([len(idxs) for idxs in idx_batch])
		# Transpose batch
		input_batch = torch.LongTensor(idx_batch).transpose(0, 1)
		input_batch = input_batch.to(device)
		lengths = lengths.to(device)
		# Decode sentence with searcher
		searcher = GreedySearchDecoder(self.text_data, self.encoder, self.decoder)
		tokens, scores = searcher(input_batch, lengths, self.text_data.max_length)
		# Map indexes back to words
		output_words = [self.text_data.idx2word[token.item()] for token in tokens]

		output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
		response = ' '.join(output_words)
		return response

	def evaluateInput(encoder, decoder, searcher, text_data):
	    input_sentence = ''
	    while(1):
	        try:
	            # Get input sentence
	            input_sentence = input('> ')
	            # Check if it is quit case
	            if input_sentence == 'q' or input_sentence == 'quit': break
	            # Normalize sentence
	            input_sentence = normalize_string(input_sentence)
	            # Evaluate sentence
	            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
	            # Format and print response sentence
	            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
	            print('Bot:', ' '.join(output_words))

	        except KeyError:
	            print("Error: Encountered unknown word.")