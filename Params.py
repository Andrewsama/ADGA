import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--batch', default=4096, type=int, help='batch size')
	parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
	parser.add_argument('--epoch', default=1000, type=int, help='number of epochs')
	parser.add_argument('--latdim', default=32, type=int, help='embedding size')
	parser.add_argument('--topk', default=[20,40], type=int, help='K of top K')
	parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
	# parser.add_argument("--ib_reg", type=float, default=0.1, help='weight for information bottleneck')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default=-1, type=int, help='indicates which gpu to use')
	parser.add_argument('--lambda0', type=float, default=1e-4, help='weight for L0 loss on laplacian matrix.')
	parser.add_argument("--eps", type=float, default=1e-3)


	parser.add_argument("--res", type=float, default=0, help="the proportion of truncated attention retained")
	parser.add_argument("--res_layer", type=float, default=0.3, help="the proportion of front-layer attention retained")  # NOTE:0
	parser.add_argument("--thread", type=float, default=0.2, help="Tthreshold of edge retained")
	parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
	parser.add_argument('--ssl_reg', default=0.1, type=float, help='weight for contrative learning')
	parser.add_argument("--seed", type=int, default=2024, help="random seed")
	parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
	parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument("--gcn_weight", type=int, default=1, help="gcn")
	parser.add_argument("--gat_weight", type=int, default=0, help="gat")

	return parser.parse_args()
args = ParseArgs()
