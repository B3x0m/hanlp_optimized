import json
import torch
from typing import List
from phrasetree.tree import Tree
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model.con.con_model import CRFConstituencyModel, CRFConstituencyDecoder
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbeddingModule


def build_tree(tokens: List[str], sequence):
    if not tokens:
        return Tree('TOP', [])
    
    tree = Tree('TOP', [Tree('_', [t]) for t in tokens])
    root = tree.label()
    leaves = [subtree for subtree in tree.subtrees() if not isinstance(subtree[0], Tree)]

    def track(node):
        i, j, label = next(node)
        if j == i + 1:
            children = [leaves[i]]
        else: 
            children = track(node) + track(node)
        
        if label.endswith('|<>'):
            return children
        
        labels = label.split('+')
        tree = Tree(labels[-1], children)
        for label in reversed(labels[:-1]):
            tree = Tree(label, [tree])
        return [tree]

    return Tree(root, track(iter(sequence)))

def process_sentence(sentence):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx_to_token = vocab["chart"]["idx_to_token"]

    batch = transform({"token": sentence})
    token_token_span = batch['token_token_span']
    max_len = max(len(span) for span in token_token_span)
    padded_spans = [span + [0] * (max_len - len(span)) for span in token_token_span]
    batch['token_token_span'] = padded_spans
    batch = {
        'token': [batch['token']],
        '_idx_': [0],
        'token_length': torch.tensor([len(batch['token'])]),
        'token_': [[token.capitalize() if token.isalpha() else token for token in batch['token']]],
        'token_input_ids': torch.tensor([batch['token_input_ids']]),
        'token_token_span': torch.tensor([batch['token_token_span']])
    }
    # print(batch)
    s_span, s_label = model(batch)

    tokens = batch.get('token_', None)
    if tokens is None:
        tokens = batch['token']
    tokens = [x[1:-1] for x in tokens]
    
    offset=1
    lens = batch['token_length'] - offset
    seq_len = lens.max()
    mask = lens.new_tensor(range(seq_len)) < lens.view(-1, 1, 1)
    mask = mask & mask.new_ones(seq_len, seq_len).triu_(1)
    span_probs = None
    if mask.any().item():
        if span_probs is None:
            s_span = decoder.crf(s_span, mask, mbr=True)
    
    chart_preds = decoder.decode(s_span, s_label, mask)
    result = [build_tree(token,[(i, j, idx_to_token[label]) for i, j, label in chart])
                for token, chart in zip(tokens, chart_preds)]
    return result

vocab_file = "./model/con/vocabs-electra.json"   # vocab-ernie.json
tokenizer_path = "./tokenizer"                   # ./ernie-tokenizer
config_file = "./model/con/config-electra.json"  # config-ernie.json
filename = "./model/con/model-electra.pt"        # model-ernie.pt

config_trans = AutoConfig.from_pretrained(config_file)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
pre_model = AutoModel.from_config(config=config_trans)
encoder = ContextualWordEmbeddingModule("token", transformer=pre_model, transformer_tokenizer=tokenizer, word_dropout=.1)
decoder = CRFConstituencyDecoder(n_labels=260, # 1353       for ernie_gram
                 n_hidden=256,                 # 768
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33)

transform = TransformerSequenceTokenizer(tokenizer_path,
                                        'token', 
                                        cls_is_bos=True,
                                        truncate_long_sequences=False)

model = CRFConstituencyModel(encoder, decoder)
model.load_state_dict(torch.load(filename, map_location='cpu', weights_only=True), strict=False)  # device etc...
model.eval()

sentence = ["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"]
result = process_sentence(sentence)

print(result)

for tree in result:
    print(tree.pretty_print())
