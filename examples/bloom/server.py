import argparse
import logging
from typing import Optional
import torch
import uvicorn

from energonai import QueueFullError, launch_engine
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from batch import BatchManagerForGeneration
from transformers import AutoTokenizer

TP_TARGET = [
    'mlp',
    'self_attention.dense',
    'self_attention.query_key_value',
    'word_embeddings.weight'
]  # 'self_attention.attention_dropout',


class GenerationTaskReq(BaseModel):
    prompt: str = Field(min_length=1)

    max_new_tokens: int = Field(gt=0, le=256)
    min_new_tokens: int = Field(gt=0, le=256)

    top_k: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    temperature: Optional[float] = Field(default=None)
    length_penalty: Optional[float] = Field(default=1.0)

    do_sample: Optional[bool] = Field(default=False)
    num_beams: Optional[int] = Field(default=1, gt=0, le=16)
    num_return_sequences: Optional[int] = Field(default=1, le=16, ge=1)
    num_beam_groups: Optional[int] = Field(default=1, ge=1, le=16)


app = FastAPI()


@app.post('/generation')
async def generate(data: GenerationTaskReq, request: Request):
    logger.info(
        f'{request.client.host}:{request.client.port} - "{request.method} {request.url.path}" - {data}'
    )

    input_tokens = tokenizer.encode_plus(data.prompt, return_tensors="pt", padding=True)
    generation_config = {}
    for field, value in data.dict().items():
        if field != "prompt":
            generation_config[field] = value
    generation_config
    try:
        uid = id(data)
        engine.submit(uid, input_tokens)
        outputs = await engine.wait(uid)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
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
    def __init__(self, model: torch.nn.Module):
        super(WrapCallModule, self).__init__()
        self.model = model

    def forward(self, **generate_kwargs):
        input_ids_batch = generate_kwargs["input_ids"]
        attention_mask_batch = generate_kwargs["attention_mask"]
        generate_kwargs["input_ids"] = torch.cat(input_ids_batch, 0)
        generate_kwargs["attention_mask"] = torch.cat(attention_mask_batch, 0)
        return self.model.generate(**generate_kwargs)


def model_fn(**model_kwargs):
    from utils import run

    tp = model_kwargs['tp'] != 1
    use_int8 = model_kwargs['dtype'] == "int8"
    from_pretrain = model_kwargs['random_init'] is False
    data_path = model_kwargs['name']
    size = model_kwargs['size']

    model = run(tp=tp, from_pretrain=from_pretrain, data_path=data_path, use_int8=use_int8, size=size)
    return WrapCallModule(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="Name path", required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--master_host', default='localhost')
    parser.add_argument('--master_port', type=int, default=19991)
    parser.add_argument('--rpc_port', type=int, default=19981)
    parser.add_argument('--max_batch_size', type=int, default=1)
    parser.add_argument('--pipe_size', type=int, default=1)
    parser.add_argument('--queue_size', type=int, default=10)
    parser.add_argument('--http_host', default='0.0.0.0')
    parser.add_argument('--http_port', type=int, default=7070)
    parser.add_argument('--max_sequence_length', type=int, default=512)
    parser.add_argument('--dtype', type=str, help="module dtype", default="fp16", choices=["fp16", "int8"])
    args = parser.parse_args()
    print_args(args)

    model_kwargs = {
        'name': args.name,
        'dtype': args.dtype,
        'random_init': False,
        'tp': args.tp,
        'size': None,
    }

    logger = logging.getLogger(__name__)
    tokenizer = AutoTokenizer.from_pretrained(args.name)

    engine = launch_engine(
        tp_world_size=args.tp,
        pp_world_size=1,
        master_host=args.master_host,
        master_port=args.master_port,
        rpc_port=args.rpc_port,
        model_fn=model_fn,
        batch_manager=BatchManagerForGeneration(
            max_batch_size=args.max_batch_size,
            pad_token_id=tokenizer.pad_token_id,
        ),
        pipe_size=args.pipe_size,
        queue_size=args.queue_size,
        **model_kwargs,
    )

    print("engine start")
    config = uvicorn.Config(app, host=args.http_host, port=args.http_port)
    server = uvicorn.Server(config=config)
    server.run()
