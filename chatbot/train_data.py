"""
Prepares the data in batches for training the model.
"""

import random
import torch
import itertools


class TrainData:
	"""
	This class transforms the data into batches 
	to be fed into the model.
	"""
	
	def __init__ (self, text_data):
		self.text_data = text_data

	def idx_from_sentence(self, sentence):
		"""
		Converts sentence to their indexs.

		Parameters:
			sentence (String): A sentence.

		Returns:
			List: A list of indexs.
		"""
		return [self.text_data.word2idx[word] for word in sentence.split(' ')] + [self.text_data.eos_token]

	def zero_padding(self, lst, fill_value=None):
		"""
		Transposes a batch to shape (max_length, batch_size) so indexing 
		across the first dimension returns a time step across all sentences
		in the batch.

		Parameters:
			lst (list): The batch of data.
			fill_value (int): Pad token to equalize sequence lengths in batch.

		Returns:
			List: A list of indexs.

		"""
		if fill_value == None:
			fill_value = self.text_data.pad_token
		return list(itertools.zip_longest(*lst, fillvalue=fill_value))

	def binary_matrix(self, lst, pad_token=None):
		"""
		Converts Matrix to binary where pad token is 0
		and all other values are 1.

		Parameters:
			lst (list): The batch of data.
			value (int): Pad token.

		Returns:
			List: A binary list.
		"""
		if pad_token == None:
			pad_token = self.text_data.pad_token
		m = []
		for i, seq in enumerate(lst):
			m.append([])
			for token in seq:
				if token == pad_token:
					m[i].append(0)
				else:
					m[i].append(1)
		return m

	def input_var(self, lst):
		"""
		Returns padded input sequence tensor and lenths.

		Parameters:
			lst (list): The batch of data.

		Returns:
			padVar (torch.long): Padded sequence tensor. 
			lengths (torch.tensor): Lengths of sequences.
		"""
		idxs_batch = [self.idx_from_sentence(sentence) for sentence in lst]
		lengths = torch.tensor([len(idxs) for idxs in idxs_batch])
		pad_list = self.zero_padding(idxs_batch)
		pad_var = torch.LongTensor(pad_list)
		return pad_var, lengths

	def output_var(self, lst):
		"""
		Returns padded target sequence tensor, padding mask and 
		max target length.

		Parameters:
			lst (list): The batch of target data.

		Returns:
			pad_var (torch.long): Padded sequence tensor.
			mask (torch.ByteTensor): Binary mask tensor.  
			max_target_length (int): Max length of sequence.
		"""	
		idxs_batch = [self.idx_from_sentence(sentence) for sentence in lst]
		max_target_length = max([len(idxs) for idxs in idxs_batch])
		pad_list = self.zero_padding(idxs_batch)
		mask = self.binary_matrix(pad_list)
		mask = torch.ByteTensor(mask)
		pad_var = torch.LongTensor(pad_list)
		return pad_var, mask, max_target_length

	def batch_2_train_data(self, pair_batch):
		"""
		Takes query/response pairs and returns input/target tensors.

		Parameters:
			pair_batch (list): List of query/response pairs.

		Returns:
			inp (torch.long): Padded sequence tensor.
			lengths (torch.tensor): Lengths of sequences.
			output (torch.long): Padded sequence tensor.
			mask (torch.ByteTensor): Binary mask tensor. 
			max_target_length (int): Max length of sequence.

		"""
		pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
		input_batch, output_batch = [], []
		for pair in pair_batch:
			input_batch.append(pair[0])
			output_batch.append(pair[1])
		inp, lengths = self.input_var(input_batch)
		output, mask, max_target_length = self.output_var(output_batch)
		return inp, lengths, output, mask, max_target_length



