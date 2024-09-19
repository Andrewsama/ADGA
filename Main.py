import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict
import os
from copy import deepcopy
import scipy.sparse as sp
import random

class Coach:
	def __init__(self, handler):
		self.handler = handler		# DataHandler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save, k=0):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			if name == 'Train':
				val = reses[metric]
				ret += '%s = %.4f, ' % (metric, val)
				tem = name + metric
				if save and tem in self.metrics:
					self.metrics[tem].append(val)
			else:
				val = reses[metric][k]
				ret += '%s@%d = %.4f, ' % (metric, args.topk[k], val)
				tem = name + metric[k]
				if save and tem in self.metrics:
					self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		# self.prepareModel()
		# log('Model Prepared')
		topk_num = len(args.topk)
		recallMax = [0] * topk_num
		ndcgMax = [0] * topk_num
		bestEpoch = [0] * topk_num

		stloc = 0
		log('Model Initialized')
		self.model = Model().cuda()
		self.R = self.handler.torchBiAdj
		self.current_adj = deepcopy(self.R).cuda()

		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		for ep in range(stloc, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0) # True
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				for k in range(topk_num):
					if reses['Recall'][k] > recallMax[k]:
						recallMax[k] = reses['Recall'][k]
						ndcgMax[k] = reses['NDCG'][k]
						bestEpoch[k] = ep
					log(self.makePrint('Test', ep, reses, tstFlag, k=k))
			print()
			self.update_current_adj()

		for k in range(topk_num):
			print('Best epoch', bestEpoch[k], ':', bestEpoch[k], ', Recall@', args.topk[k] ,':', recallMax[k], ', NDCG@', args.topk[k] ,': ', ndcgMax[k])

		# TODO:更方便复制
		print()
		print(bestEpoch[0])
		print(bestEpoch[1])
		for k in range(topk_num):
			print(recallMax[k])
			print(ndcgMax[k])
		with open('result_beer.txt', 'a+') as fn:
			fn.write(str(args.res_layer)+' '+str(args.thread)+'\n')
			fn.write(str(bestEpoch[0])+'\n')
			fn.write(str(bestEpoch[1])+'\n')
			for k in range(topk_num):
				fn.write(str(recallMax[k])+'\n')
				fn.write(str(ndcgMax[k])+'\n')
			fn.write('\n')


	def gatemb(self, adj):
		self.new_adj, emb = self.model.forward_gat(adj)
		return emb

	def gcnemb(self, adj):
		emb = self.model.forward_gcn(adj)
		return emb

	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		loss, loss_cl, loss_bpr, loss_reg = 0, 0, 0, 0
		steps = trnLoader.dataset.__len__() // args.batch

		for i, tem in enumerate(trnLoader):
			gat_emb = self.gatemb(self.current_adj)
			gcn_emb = self.gcnemb(self.R.cuda())

			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			loss_cl = self.model.loss_graphcl(gat_emb, gcn_emb, ancs, poss).mean() * args.ssl_reg
			loss_bpr = self.model.loss_graphbpr(gat_emb, gcn_emb, ancs, poss, negs)
			loss_reg = calcRegLoss(self.model) * args.reg
			loss = loss_cl + loss_bpr + loss_reg

			self.opt.zero_grad()
			loss.backward()
			self.opt.step()

			log('Step %d/%d: loss_bpr : %.3f ; loss_cl : %.3f ; loss_reg : %.3f' % (
				i, 
				steps,
				loss_bpr,
                loss_cl,
                loss_reg
				), save=False, oneline=True)

		ret = dict()
		ret['BPR Loss'] = loss_bpr / steps
		ret['CL Loss'] = loss_cl / steps
		ret['Reg Loss'] = loss_reg / steps

		return ret

	def testEpoch(self):
		with torch.no_grad():
			tstLoader = self.handler.tstLoader
			args_topk = args.topk
			epRecall = [0] * len(args_topk)
			epNdcg = [0] * len(args_topk)
			i = 0
			num = tstLoader.dataset.__len__()
			steps = num // args.tstBat
			for usr, trnMask in tstLoader:
				i += 1
				usr = usr.long().cuda()
				trnMask = trnMask.cuda()
				gat_emb = self.gatemb(self.current_adj)
				gcn_emb = self.gcnemb(self.R.cuda())
				usrEmbeds, itmEmbeds = self.model.get_embeds(gat_emb, gcn_emb)

				# usrEmbeds, itmEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)
				allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
				topLocs = []
				for k in args_topk:
					topLocs.append(torch.topk(allPreds, k)[1])
				recall = []
				ndcg = []
				for k in range(len(topLocs)):
					recall_tmp, ndcg_tmp = self.calcRes(topLocs[k].cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, args_topk[k])
					recall.append(recall_tmp)
					ndcg.append(ndcg_tmp)

				for k in range(len(args_topk)):
					epRecall[k] += recall[k]
					epNdcg[k] += ndcg[k]
				for k in range(len(args_topk)):
					log('Steps %d/%d: recall@%d = %.2f, ndcg@%d = %.2f          ' % (i, steps, args_topk[k], recall[k], args_topk[k], ndcg[k]), save=False, oneline=True)
			ret = dict()

			ret['Recall'] = [k / num for k in epRecall]
			ret['NDCG'] = [k / num for k in epNdcg]
		return ret

	def update_current_adj(self):
		result = [] # todo
		with torch.no_grad():
			row, col = self.new_adj._indices().cpu().numpy()
			value = self.new_adj._values().cpu().numpy()
			ori_value = self.new_adj._values().cpu().numpy()
			rev_value = {}
			for i, j, v in zip(col, row, value):
				rev_value[(i, j)] = v
			for j in range(len(col)):
				# tmp = (max(value[j], rev_value[(col[j], row[j])]) + args.res * ori_value[j]) / (1 + args.res)
				tmp = max(value[j], rev_value[(col[j], row[j])])
				result.append(tmp) # todo
				if tmp > args.thread:
					value[j] = 1
				else:
					row[j], col[j], value[j] = -1, -1, 0
			row = row[row != -1]
			col = col[col != -1]
			value = value[value != 0]
			csr_adj = sp.coo_matrix((value, (row, col)), self.new_adj.shape).tocsr()
			degree = np.array(csr_adj.sum(axis=-1))
			dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
			dInvSqrt[np.isinf(dInvSqrt)] = 0.0
			dInvSqrtMat = sp.diags(dInvSqrt)
			coo_adj = csr_adj.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
			idxs = torch.from_numpy(np.vstack([coo_adj.row, coo_adj.col]).astype(np.int64))
			vals = torch.from_numpy(coo_adj.data.astype(np.float32))
			shape = torch.Size(coo_adj.shape)
			self.current_adj = torch.sparse.FloatTensor(idxs, vals, shape).cuda()
		# with open('att.txt' ,'a+') as fn:
		# 	np.savetxt(fn, result, delimiter=",")

	def calcRes(self, topLocs, tstLocs, batIds, argk):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, argk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	with torch.cuda.device(args.gpu):
		logger.saveDefault = True
		seed_it(args.seed)

		log('Start')
		handler = DataHandler()
		handler.LoadData()
		log('Load Data')

		coach = Coach(handler)
		coach.run()