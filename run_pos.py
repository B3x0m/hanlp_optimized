import onnxruntime as ort
import numpy as np
import time
import torch
import json
from concurrent.futures import ThreadPoolExecutor
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer


def decode_output(x):
    with open(vocab_file, "r") as f:
        vocabs = json.load(f)
    tag_vocab = vocabs["tag"]["idx_to_token"]
    result = []
    for idx in x[0].tolist():
        result.append(tag_vocab[idx])
    return result

def process_sentence(sentence):
    inputs = transform({'token': sentence})
    lens = len(inputs['token'])

    token_token_span = inputs['token_token_span']
    max_len = max(len(span) for span in token_token_span)
    padded_spans = [span + [0] * (max_len - len(span)) for span in token_token_span]
    inputs['token_token_span'] = padded_spans

    text = inputs['token']
    input_ids = inputs['token_input_ids']
    lens = np.array([lens], dtype=np.int64)
    input_ids = np.array([input_ids], dtype=np.int64)
    token_span = np.array([inputs['token_token_span']], dtype=np.int64)
    input_names = [input.name for input in session.get_inputs()]
    
    logits, mask = session.run(None, {
        input_names[0]: lens,       # lens
        input_names[1]: input_ids,  # input_ids
        input_names[2]: token_span  # token_span
    })
    
    predictions = logits.argmax(-1)
    print(predictions)
    preds = decode_output(predictions)

    return preds

vocab_file = "./model/pos/vocabs-pku.json"  #vocabs-ctb.json
tokenizer_path = "./tokenizer"
onnx_path = "./model/yolo-pos-pku.onnx"     #yolo-pos-ctb.onnx

so = ort.SessionOptions()
so.intra_op_num_threads=1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']) # CUDAExecutionProvider
session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
transform = TransformerSequenceTokenizer(tokenizer_path, 'token',
                                        ret_token_span=True,
                                        ret_subtokens=False,
                                        ret_mask_and_type=False,
                                        ret_subtokens_group=False,
                                        ret_prefix_mask=False,)


sentence = ["阿婆主", "来到", "北京", "立方庭", "参观", "自然", "语义", "科技", "公司", "。"]

results = process_sentence(sentence)
print(results)