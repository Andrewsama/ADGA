from torch import nn
import torch.nn.functional as F
import torch
from Params import args
from copy import deepcopy
import numpy as np
import math
import scipy.sparse as sp
from Utils.Utils import contrastLoss, calcRegLoss, pairPredict
import time
import torch_sparse

init = nn.init.xavier_uniform_

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gatLayers = nn.Sequential(*[GATLayer(args.latdim) for i in range(args.gnn_layer)])# [GCNLayer(), GCNLayer(), ...]
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])# [GCNLayer(), GCNLayer(), ...]
		self.conLayers = nn.Linear(args.latdim * 2, args.latdim)

	def forward_gat(self, adj):
		iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]
		for gat in self.gatLayers:
			adj, embeds = gat(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)

		return adj, mainEmbeds

	def forward_gcn(self, adj):
		iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)

		return mainEmbeds

	def loss_graphcl(self, x1, x2, users, items):
		T = args.temp
		user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)
		user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)

		user_embeddings1 = F.normalize(user_embeddings1, dim=1)
		item_embeddings1 = F.normalize(item_embeddings1, dim=1)
		user_embeddings2 = F.normalize(user_embeddings2, dim=1)
		item_embeddings2 = F.normalize(item_embeddings2, dim=1)

		user_embs1 = F.embedding(users, user_embeddings1)
		item_embs1 = F.embedding(items, item_embeddings1)
		user_embs2 = F.embedding(users, user_embeddings2)
		item_embs2 = F.embedding(items, item_embeddings2)

		all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
		all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

		all_embs1_abs = all_embs1.norm(dim=1)
		all_embs2_abs = all_embs2.norm(dim=1)
	
		sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs, all_embs2_abs)
		sim_matrix = torch.exp(sim_matrix / T)
		pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
		loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
		loss = - torch.log(loss)

		return loss

	def loss_graphbpr(self, gat_emb, gcn_emb, ancs, poss, negs):
		# emb = self.conLayers(torch.concat([gat_emb, gcn_emb], dim=-1))
		# user_emb, item_emb = torch.split(emb, [args.user, args.item], dim=0)
		# ancEmbeds = user_emb[ancs]
		# posEmbeds = item_emb[poss]
		# negEmbeds = item_emb[negs]
		# scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		# bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch

		user_gat, item_gat = torch.split(gat_emb, [args.user, args.item], dim=0)
		ancEmbeds1 = user_gat[ancs]
		posEmbeds1 = item_gat[poss]
		negEmbeds1 = item_gat[negs]
		# scoreDiff1 = pairPredict(ancEmbeds1, posEmbeds1, negEmbeds1)
		# bprLoss1 = - (scoreDiff1).sigmoid().log().sum() / args.batch

		user_gcn, item_gcn = torch.split(gcn_emb, [args.user, args.item], dim=0)
		ancEmbeds = user_gcn[ancs]
		posEmbeds = item_gcn[poss]
		negEmbeds = item_gcn[negs]
		# scoreDiff2 = pairPredict(ancEmbeds2, posEmbeds2, negEmbeds2)
		# bprLoss2 = - (scoreDiff2).sigmoid().log().sum() / args.batch

		# ancEmbeds = (args.gat_weight*ancEmbeds1+args.gcn_weight*ancEmbeds2)/(args.gat_weight+args.gcn_weight)
		# posEmbeds = (args.gat_weight*posEmbeds1+args.gcn_weight*posEmbeds2)/(args.gat_weight+args.gcn_weight)
		# negEmbeds = (args.gat_weight*negEmbeds1+args.gcn_weight*negEmbeds2)/(args.gat_weight+args.gcn_weight)
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch

		return bprLoss #+ bprLoss1 + bprLoss2

	def get_embeds(self, gat_emb, gcn_emb):
		# emb = self.conLayers(torch.concat([gat_emb, gcn_emb], dim=-1))
		# emb = (args.gat_weight*gat_emb + args.gcn_weight * gcn_emb)/ (args.gat_weight+args.gcn_weight)

		user_emb, item_emb = torch.split(gcn_emb, [args.user, args.item], dim=0)
		return user_emb, item_emb


class GATLayer(nn.Module):
	def __init__(self, hidden):
		super(GATLayer, self).__init__()
		# self.weight = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.weight_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.weight_2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.attention = nn.Sequential(nn.Linear(2 * hidden, 1))

	def forward(self, adj, embeds, flag=True):
		ind = adj._indices()
		self.row = ind[0, :]
		self.col = ind[1, :]
		f1_feat = embeds[self.row]
		f2_feat = embeds[self.col]
		input = torch.concat([self.weight_1(f1_feat), self.weight_2(f2_feat)], dim=1)
		att = self.attention(input)
		exp_att = torch.exp(att)
		exp_att_s = torch.squeeze(exp_att) # e^attention (434248,)
		tmp_adj = torch.sparse.FloatTensor(ind, exp_att_s, adj.shape)

		rowsum = torch.sparse.sum(tmp_adj, dim=-1).to_dense() + 1e-6 # sum(e^attention) (69534,)
		rowsum_T = torch.unsqueeze(torch.pow(rowsum, -1), dim=1) # 1 / sum(e^attention) (69534,1)
		edge_rowsum = rowsum_T[self.row]	# 1 / sum(e^attention) (434248, 1)
		# a = args.res * adj._values()
		# b = torch.squeeze(torch.mul(exp_att, edge_rowsum))
		# c = a+b
		# d = c / (1 + args.res)
		values = (torch.squeeze(torch.mul(exp_att, edge_rowsum)) + args.res_layer * adj._values()) / (1 + args.res_layer)
		support = torch.sparse.FloatTensor(ind, values, adj.shape).coalesce()

		# if (flag):
		# 	return torch.spmm(adj, embeds)
		# else:
		# 	return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)
		return support, torch_sparse.spmm(ind, values, adj.shape[0], adj.shape[1], embeds)

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds, flag=True):
		if (flag):
			return torch.spmm(adj, embeds)
		else:
			return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)

