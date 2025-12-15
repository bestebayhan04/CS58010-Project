
import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "your_data_root"


args.dataset = ["CS"]
args.simulation_mode = "subgraph_fl_louvain"
args.num_clients = 10


if True:
    args.fl_algorithm = "hcfl_plus"
    args.model = ["gcn"]
else:
    args.fl_algorithm = "fedproto"
    args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"] # choose multiple gnn models for model heterogeneity setting.

args.metrics = ["accuracy"]
args.num_clusters = 3


trainer = FGLTrainer(args)

trainer.train()

#import torch
#from torch_geometric.nn import GCNConv
#
#print("CUDA:", torch.cuda.is_available())
#
#conv = GCNConv(16, 16).cuda()
#x = torch.randn(100, 16).cuda()
#edge_index = torch.randint(0, 100, (2, 500)).cuda()
#
#out = conv(x, edge_index)
#print("Output device:", out.device)

