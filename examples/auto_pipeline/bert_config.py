from bert import BertEmbedding1D, BertTransformerLayer1D, bert_8B, bert_175B, bert_large, bert_small, bert_xl
from bert_server import launch_engine


model_class = bert_8B
model_type = "bert"
engine_server = launch_engine


# parallel
tp_init_size = 2
pp_init_size = 2
auto_pp = True
LeafSet = set([BertTransformerLayer1D, BertEmbedding1D])



host = "127.0.0.1"
port = 29400
half = False
server_host = "127.0.0.1"
server_port = 8010
log_level = "info"
backend = "nccl"