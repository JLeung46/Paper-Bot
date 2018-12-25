import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def maskNLLLoss(inp, target, mask):
	"""
	Calculates the average negative log likelihood
	of the elements that correspond to a 1 in the mask.
	""" 
	nTotal = mask.sum()
	crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
	loss = crossEntropy.masked_select(mask).mean()
	loss = loss.to(device)
	return loss, nTotal.item()
