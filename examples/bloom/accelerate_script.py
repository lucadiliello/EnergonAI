import argparse
import logging
import random
from typing import Optional
import torch
import uvicorn
import colossalai
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ShardSpec, ComputeSpec, ComputePattern, ColoParameter, ProcessGroup

from energonai import QueueFullError, launch_engine
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from batch import BatchManagerForGeneration
from cache import ListCache, MissCacheError
from transformers import AutoTokenizer, BloomForCausalLM
from transformers import BloomConfig

class GenerationTaskReq(BaseModel):
    max_new_tokens: int = Field(gt=0, le=256, example=64)
    prompt: str = Field(
        min_length=1, example='Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:')
    top_k: Optional[int] = Field(default=None, gt=0, example=50)
    top_p: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.5)
    greedy: Optional[bool] = False


app = FastAPI()


@app.post('/generation')
async def generate(data: GenerationTaskReq, request: Request):
    logger.info(
        f'{request.client.host}:{request.client.port} - "{request.method} {request.url.path}" - {data}')
    key = (data.prompt, data.max_new_tokens)
    try:
        if cache is None:
            raise MissCacheError()
        outputs = cache.get(key)
        outputs = random.choice(outputs)
        logger.info('Cache hit')
    except MissCacheError:
        input_tokens = tokenizer.encode_plus(data.prompt, return_tensors="pt", padding=True)
        input_tokens['max_new_tokens'] = data.max_new_tokens
        input_tokens['top_k'] = data.top_k
        input_tokens['top_p'] = data.top_p
        try:
            uid = id(data)
            engine.submit(uid, input_tokens)
            outputs = await engine.wait(uid)
            outputs = tokenizer.decode(outputs, skip_special_tokens=True)
            if cache is not None:
                cache.add(key, outputs)
        except QueueFullError as e:
            raise HTTPException(status_code=406, detail=e.args[0])

    return {'text': outputs}


@app.on_event("shutdown")
async def shutdown(*_):
    engine.shutdown()
    server.should_exit = True
    server.force_exit = True
    await server.shutdown()


def print_args(args: argparse.Namespace):
    print('\n==> Args:')
    for k, v in args.__dict__.items():
        print(f'{k} = {v}')        
    
class WrapCallModule(torch.nn.Module):
    def __init__(self, model:torch.nn.Module):
        super(WrapCallModule, self).__init__()
        self.model = model
        
    def forward(self, **generate_kwargs):
        return self.model.generate(**generate_kwargs)

def model_fn(**model_kwargs):
    model_name = model_kwargs['name']
    use_tp = True

    if use_tp:
        # colossalai.launch_from_torch(config={})
        print(f'init TP world size {torch.distributed.get_world_size()}')
        with ColoInitContext(device=torch.cuda.current_device()):
            colo_model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")
        
        def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
            spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
            if param.process_group.tp_world_size() == 1:
                param.set_process_group(pg)
            param.set_tensor_spec(*spec)

        def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
            split_param_single_dim_tp1d(0, param, pg)
        pg = ProcessGroup(tp_degree=torch.distributed.get_world_size())
        for mn, module in colo_model.named_modules():
            for pn, param in module.named_parameters(recurse=False):
                # reset process group for all parameters
                param.set_process_group(pg)

                if 'dense_h_to_4h.weight' in pn or 'self_attention.query_key_value' in pn or 'mlp.dense_4h_to_h' in pn:
                    split_param_row_tp1d(param, pg)  # colmn slice
        print('initialize TP OK')
        return WrapCallModule(colo_model)
    else:
        model = BloomForCausalLM.from_pretrained(model_name, **kwargs)
        print(model.config)
        return WrapCallModule(model)
    # model config only:
    # configuration = BloomConfig(hidden_size=1024, #64
    #                             n_layer=32, #2
    #                             n_head=128, #8
    #                             ) 
    # model = BloomForCausalLM(configuration)
    
    # for name, p in model.named_parameters():
    #     print(name)
    

FIXED_CACHE_KEYS = [
    ('Question: What is the name of the largest continent on earth?\nAnswer: Asia\n\nQuestion: What is at the center of the solar system?\nAnswer:', 64),
    ('A chat between a salesman and a student.\n\nSalesman: Hi boy, are you looking for a new phone?\nStudent: Yes, my phone is not functioning well.\nSalesman: What is your budget? \nStudent: I have received my scholarship so I am fine with any phone.\nSalesman: Great, then perhaps this latest flagship phone is just right for you.', 64),
    ("English: I am happy today.\nChinese: 我今天很开心。\n\nEnglish: I am going to play basketball.\nChinese: 我一会去打篮球。\n\nEnglish: Let's celebrate our anniversary.\nChinese:", 64)
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="Name path", required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--master_host', default='localhost')
    parser.add_argument('--master_port', type=int, default=19991)
    parser.add_argument('--rpc_port', type=int, default=19981)
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--pipe_size', type=int, default=1)
    parser.add_argument('--queue_size', type=int, default=0)
    parser.add_argument('--http_host', default='0.0.0.0')
    parser.add_argument('--http_port', type=int, default=7070)
    parser.add_argument('--cache_size', type=int, default=0)
    parser.add_argument('--cache_list_size', type=int, default=1)
    args = parser.parse_args()
    print_args(args)

    num_tokens = 100
    model_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)
    model_name = args.name
    model_kwargs['name'] = model_name
    logger = logging.getLogger(__name__)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.cache_size > 0:
        cache = ListCache(args.cache_size, args.cache_list_size,
                          fixed_keys=FIXED_CACHE_KEYS)
    else:
        cache = None
    engine = launch_engine(args.tp, 1, args.master_host, args.master_port, args.rpc_port, model_fn,
                           batch_manager=BatchManagerForGeneration(max_batch_size=args.max_batch_size,
                                                                   pad_token_id=tokenizer.pad_token_id),
                           pipe_size=args.pipe_size,
                           queue_size=args.queue_size,
                           **model_kwargs)
    print("engine start")
    config = uvicorn.Config(app, host=args.http_host, port=args.http_port)
    server = uvicorn.Server(config=config)
    server.run()