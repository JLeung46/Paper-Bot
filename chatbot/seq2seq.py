"""
Builds a seq2seq model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
	"""
	This class builds the encoder.
	"""
	def __init__(self, embedding, hidden_size=500, n_layers=2, dropout=0.1):
		super(EncoderRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = embedding

		self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
						  dropout= (0 if n_layers == 1 else dropout), bidirectional=True)

	def forward(self, input_seq, input_lengths, hidden=None):
		"""
		Encoder forward pass.
		"""
		# Convert word indexes to embeddings
		embedded = self.embedding(input_seq)
		# Pack padded batch of sequences, helps optimize number of computations
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		# Forward pass through GRU
		outputs, hidden = self.gru(packed, hidden)
		# Unpack padding
		outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
		# Sum bidirectional GRU outputs
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
		return outputs, hidden

class Attn(torch.nn.Module):
	"""
	Luong attention layer.
	"""
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, "is not an appropriate attention method")
		self.hidden_size = hidden_size
		if self.method == 'general':
			self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
		elif self.method == 'concat':
			self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = torch.nn.Parameter(torch,FloatTensor(hidden_size))

	def dot_score(self, hidden, encoder_output):
		return torch.sum(hidden * encoder_output, dim=2)

	def general_score(self, hidden, encoder_output):
		energy = self.attn(encoder_output)
		return torch.sum(hidden * energy, dim=2)

	def concat_score(self, hidden, encoder_output):
		# Attention weights
		energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
			encoder_output), 2)).tanh()
		return torch.sum(self.v * energy, dim=2)

	def forward(self, hidden, encoder_outputs):
		# Calculate atttention weights (energy) based on method.
		if self.method == 'general':
			attn_energy = self.general_score(hidden, encoder_outputs)
		elif self.method == 'concat':
			attn_energy = self.concat_score(hiddden, encoder_outputs)
		elif self.method == 'dot':
			attn_energy = self.dot_score(hidden, encoder_outputs)

		# Transpose max_length and batch_size dimensions
		attn_energy = attn_energy.t()

		# Return the softmax normalized probability scores
		return F.softmax(attn_energy, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1,
		dropout=0.1):
		super(LuongAttnDecoderRNN, self).__init__()

		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = embedding
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
						  dropout=(0 if n_layers == 1 else dropout))
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.attn = Attn(attn_model, hidden_size)

	def forward(self, input_step, last_hidden, encoder_outputs):
		"""
		Parameters:
			input_step: one time step on input batch
			last_hidden: last hidden layer of GRU
			encoder_outputs: encoder model output

		Returns:
			output: softmax normalized probabilities
			hidden: final hidden state of GRU
		"""
		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)
		rnn_output, hidden = self.gru(embedded, last_hidden)
		# Calculate attention weights from GRU output
		attn_weights = self.attn(rnn_output, encoder_outputs)
		# Get new weighted sum context vector  by multiplying attn weights
		# to encoder outputs 
		context = attn_weights.bmm(encoder_outputs.transpose(0,1))
		# Concatenate weighted context vector and GRU output using Luong eq. 5
		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))
		# Predict next word using Luong eq. 6
		output = self.out(concat_output)
		output = F.softmax(output, dim=1)
		# Return output and final hidden state
		return output, hidden		

