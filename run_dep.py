import json
import torch
from model.dep.dep_model import BiaffineDependencyModel
from transformers import AutoConfig, AutoModel, AutoTokenizer
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer


config = {'average_subword': True,
        'average_subwords': False,
        'batch_size': None,
        'decay': 0.75,
        'decay_steps': 5000,
        'embed_dropout': 0.33,
        'encoder_lr': 0.0001,
        'transformer_lr': 5e-05,
        'epochs': 30,
        'epsilon': 1e-12,
        'feat': None,
        'finetune': False,
        'grad_norm': 1.0,
        'gradient_accumulation': 2,
        'hidden_dropout': 0.33,
        'layer_dropout': 0,
        'lowercase': False,
        'lr': 0.001,
        'max_sequence_length': 512,
        'min_freq': 2,
        'mlp_dropout': 0.33,
        'mu': 0.9,
        'n_embed':100,
        'n_lstm_hidden': 400,
        'n_lstm_layers': 3,
        'n_mlp_arc': 500,
        'n_mlp_rel': 100,
        'n_rels': 46,
        'nu': 0.9,
        'patience': 30,
        'pretrained_embed': None,
        'proj': True,
        'punct': True,
        'sampler_builder': {'classpath': 'hanlp.common.dataset.SortingSamplerBuilder',
                            'use_effective_tokens': False,
                            'batch_max_tokens': None,
                            'batch_size': 32},
        'scalar_mix': None,
        'secondary_encoder': None,
        'seed': 1644964061,
        'separate_optimizer': False,
        'transform': 'hanlp.common.transform.NormalizeCharacter',
        'transformer': 'hfl/chinese-electra-180g-small-discriminator',
        'transformer_hidden_dropout': None,
        'transformer_lr': 5e-05,
        'tree': True,
        'unk': '<unk>',
        'pad': '<unk>',
        'warmup_steps': 0.1,
        'weight_decay': 0, 'word_dropout': 0.1,
        'classpath': 'hanlp.components.parsers.biaffine.biaffine_dep.BiaffineDependencyParser',
        'hanlp_version': '2.1.0-beta.15'}

def decode_output(x):
    with open(vocab_file, "r") as f:
        vocabs = json.load(f)
    tag_vocab = vocabs["rel"]["idx_to_token"]
    result = []
    for idx in x[0].tolist():
        result.append(tag_vocab[idx])
    return result

def process_sentence(sentence):
    sentence = ["[CLS]"] + sentence + ["<eos>"]
    inputs = transform({'token': sentence})

    lens = torch.tensor([len(inputs['token'])])
    token_token_span = inputs['token_span']
    max_len = max(len(span) for span in token_token_span)
    padded_spans = [span + [0] * (max_len - len(span)) for span in token_token_span]
    mask_len = lens.item()
    mask_in = torch.arange(mask_len).unsqueeze(0) < lens.unsqueeze(0)
    inputs['token_span'] = padded_spans
    
    words = None
    feats = None
    input_ids=torch.tensor([inputs["input_ids"]])
    token_span=torch.tensor([inputs["token_span"]])
    logits, mask = model(words, feats, input_ids, token_span, mask_in, lens)
    logits_preds = logits[:, 1:-1, :].argmax(-1)
    mask_score = mask[:, 1:-1, :].argmax(-1)
    mask_out = mask_score.gather(-1, logits_preds.unsqueeze(-1)).squeeze(-1)
    return logits_preds, mask_out


vocab_file = "./model/dep/vocabs-ctb.json" # vocab-udc.json
tokenizer_path = "./tokenizer"
config_file = "./model/dep/config.json"
filename = "./model/dep/model.pt"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
transform = TransformerSequenceTokenizer(tokenizer_path,
                                        'token', '', 
                                        ret_token_span=True,
                                        ret_subtokens=False,
                                        ret_mask_and_type=False,
                                        ret_subtokens_group=False,
                                        ret_prefix_mask=False,)
config_trans = AutoConfig.from_pretrained(config_file)
pre_model = AutoModel.from_config(config=config_trans)
model = BiaffineDependencyModel(config, None, pre_model, tokenizer)
model.load_state_dict(torch.load(filename, map_location='cpu', weights_only=True), strict=False)


sentence = ["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"]
logits_out, mask_out = process_sentence(sentence)
mask_out = decode_output(mask_out)
print(logits_out)
print(mask_out)