from repeaters import RepeaterNetwork
from models import CNN, GNN
from gnn_env import Environment as gnnenv
from cnn_env import Environment as cnnenv

model = GNN()
env = gnnenv(GNN())
env.preview()