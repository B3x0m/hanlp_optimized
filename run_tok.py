import json
import torch
import numpy as np
from typing import Set
from trie_file import *
from dictionary import *
import onnxruntime as ort
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer


class TransformerTaggingTokenizer():

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def id_to_tags(self, ids: torch.LongTensor, lens: List[int]):
        batch = []
        with open(vocab_file, "r") as f:
            vocabs = json.load(f)
        idx_to_token = vocabs['tag']['idx_to_token']

        batch = []
        vocab = idx_to_token
        for b, l in zip(ids, lens):
            batch.append([])
            for i in b[:l]:
                batch[-1].append(vocab[i])
        return batch

    def decode_output(self, logits, mask, batch, model=None):
        output = logits
        if isinstance(output, torch.Tensor):
            output = output.tolist()
        prediction = self.id_to_tags(output, [len(x) + 2 for x in batch['token']])
        return self.tag_to_span(prediction, batch)

    def bmes_to_spans(self, tags):
        result = []
        offset = 0
        pre_offset = 0
        for t in tags[1:]:
            offset += 1
            if t == 'B' or t == 'S':
                result.append((pre_offset, offset))
                pre_offset = offset
        if offset != len(tags):
            result.append((pre_offset, len(tags)))
        return result

    def tag_to_span(self, batch_tags, batch: dict):
        spans = []
        if 'custom_words' in batch:
            if self.config.tagging_scheme == 'BMES':
                S = 'S'
                M = 'M'
                E = 'E'
            else:
                S = 'B'
                M = 'I'
                E = 'I'
            for tags, custom_words in zip(batch_tags, batch['custom_words']):
                # [batch['raw_token'][0][x[0]:x[1]] for x in subwords]
                if custom_words:
                    for start, end, label in custom_words:
                        if end - start == 1:
                            tags[start] = S
                        else:
                            tags[start] = 'B'
                            tags[end - 1] = E
                            for i in range(start + 1, end - 1):
                                tags[i] = M
                        if end < len(tags):
                            tags[end] = 'B'
        if 'token_subtoken_offsets_group' not in batch:  # only check prediction on raw text for now
            # Check cases that a single char gets split into multiple subtokens, e.g., ‥ -> . + .
            for tags, subtoken_offsets in zip(batch_tags, batch['token_subtoken_offsets']):
                offset = -1  # BERT produces 'ᄒ', '##ᅡ', '##ᆫ' for '한' and they share the same span
                prev_tag = None
                for i, (tag, (b, e)) in enumerate(zip(tags, subtoken_offsets)):
                    if b < offset:
                        if prev_tag == 'S':
                            tags[i - 1] = 'B'
                        elif prev_tag == 'E':
                            tags[i - 1] = 'M'
                        tags[i] = tag = 'M'
                    offset = e
                    prev_tag = tag
        for tags in batch_tags:
            spans.append(self.bmes_to_spans(tags))
        return spans

def process_sentence(sentence):
    inputs = transform({'token': [sentence]})
    lens = len(inputs['token_input_ids'])
    features = [inputs[k] for k in inputs]
    
    text, token_span, input_ids = features
    lens = np.array([lens], dtype=np.int64)
    input_ids = np.array(input_ids, dtype=np.int64)

    input_names = [input.name for input in session.get_inputs()]

    logits, mask = session.run(None, {
        input_names[0]: lens,       # lens
        input_names[1]: input_ids   # input_ids
    })
    
    predictions = logits[:, 1:-1, :].argmax(-1)
    
    preds = toke.decode_output(predictions, mask, inputs)
    token_ids = inputs['token_input_ids'][1:-1]

    result = []
    for pred in preds:
        tokens = []  
        for start, end in pred:
            span_token_ids = token_ids[start:end]
            decoded_tokens = tokenizer.decode(span_token_ids)
            decoded_tokens = decoded_tokens.replace(' ', '')
            tokens.append(decoded_tokens)
        result.append(tokens)
    return tokens


vocab_file = "./model/tok/vocabs.json"
tokenizer_path = "./tokenizer"
onnx_path = "./model/yolo.onnx"

toke = TransformerTaggingTokenizer()
so = ort.SessionOptions()
so.intra_op_num_threads=1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']) # CUDAExecutionProvider
session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
transform = TransformerSequenceTokenizer(tokenizer_path, 'token',
                                        ret_subtokens=True,
                                        ret_subtokens_group=True,
                                        ret_token_span=False,
                                        use_fast=True,)

sentences = [
    '第293章 佐助IF线（14）阴谋，计划',
'旅祸们在瀞灵廷大闹，搅的死神们不得安宁。',
'他们人数虽少，可惹出来的麻烦，却意外的让人恐慌。',
'两位队长被击败！',
]

results = [process_sentence(sentence) for sentence in sentences]
print("hanlp_xom: ", results)
