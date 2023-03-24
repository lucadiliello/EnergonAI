from typing import Any, Deque, Hashable, List, Tuple

from encoding import batch_encode_with_prefix_and_postfix, decode_batch
from transformers.tokenization_utils import PreTrainedTokenizerBase

from energonai import BatchManager, SubmitEntry, TaskEntry


class BatchManagerForGeneration(BatchManager):

    def __init__(
        self,
        max_batch_size: int = 1,
        tokenizer: PreTrainedTokenizerBase = None,
        max_sequence_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size

    def make_batch(self, queue: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        # generation config will be the same for all elements in batch
        generation_config = queue[0].data['generation_config']
        prefix = queue[0].data['prefix']
        postfix = queue[0].data['postfix']

        # initialize uids and batch lists
        uids = []
        batch = []

        # increase the batch size until:
        # 1) there are elements in the queue
        # 2) we are not exceeding max batch size
        # 3) generation configs are equal
        # 4) prefix or postfix are equal
        while (len(queue) > 0) and (len(batch) < self.max_batch_size) and (
            queue[0].data['generation_config'] == generation_config
        ) and (
            queue[0].data['prefix'] == prefix
        ) and (
            queue[0].data['postfix'] == postfix
        ):
            # get new element from queue and remove generation config
            e = queue.popleft()

            # add prompts to batch
            batch.append(e.data['prompt'])
            uids.append(e.uid)

        inputs = batch_encode_with_prefix_and_postfix(
            batch,
            prefix=prefix,
            postfix=postfix,
            max_sequence_length=self.max_sequence_length,
            tokenizer=self.tokenizer,
        )
        inputs['generation_config'] = generation_config

        # return data for model and additional info that will be used for decoding
        return TaskEntry(tuple(uids), inputs), {
            'num_return_sequences': generation_config.num_return_sequences,
            'input_length': inputs['input_ids'].shape[1],
        }

    def split_batch(
        self, task_entry: TaskEntry, num_return_sequences: int = None, input_length: int = None
    ) -> List[Tuple[Hashable, Any]]:
        output_sentences = decode_batch(
            task_entry.batch,
            self.tokenizer,
            num_return_sequences=num_return_sequences,
            original_input_length=input_length,
        )
        return [(uid, output_sentence) for uid, output_sentence in zip(task_entry.uids, output_sentences)]
