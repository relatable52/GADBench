import argparse
import time
from utils import *
import pandas
import os
import warnings
import pickle
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--inductive', type=int, default=0)
args = parser.parse_args()

columns = ['name']
new_row = {}
datasets = ['reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
            'elliptic', 'elliptic_lf', 'tolokers', 'questions', 'dgraphfin', 'tsocial', 'hetero/amazon', 'hetero/yelp']

train_config = {
            'device': 'cuda',
            'epochs': 200,
            'patience': 50,
            'metric': 'AUPRC',
            'inductive': bool(args.inductive)
        }

dataset_name = "elliptic_lf"
model = "GCN"
data = Dataset(dataset_name)
model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
torch.cuda.empty_cache()
print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, 1))
data.split(args.semi_supervised, 1)
seed = seed_list[1]
set_seed(seed)
train_config['seed'] = seed
detector = model_detector_dict[model](train_config, model_config, data)
st = time.time()
print(detector.model)
test_score = detector.train()

gcn_model = detector.model
graph = detector.source_graph
h = gcn_model.get_hidden_state(graph)
print(graph.nodes())
print(h, h.shape)
print(graph.ndata['feature'])

