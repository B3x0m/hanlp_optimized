import json
import torch
import importlib
from torch import nn
from collections import defaultdict
from typing import Union, List, Dict, Any, Tuple
from transformers import AutoConfig, AutoModel, AutoTokenizer

from hanlp.components.mtl.tasks import Task
from hanlp.utils.torch_util import lengths_to_mask
from hanlp.common.transform import FieldLength, TransformList
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.layers.transformers.encoder import TransformerEncoder
from hanlp.layers.transformers.pt_imports import PreTrainedTokenizer
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer

from mtl_task.pos import TransformerTagging
from mtl_task.con import CRFConstituencyParsing
from mtl_task.tag_tok import TaggingTokenization
from mtl_task.dep import BiaffineDependencyParsing
from mtl_task.tag_ner import TaggingNamedEntityRecognition
# from mtl_task.sdp import BiaffineSemanticDependencyParsing
# from mtl_task.srl.bio_srl import SpanBIOSemanticRoleLabeling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_transform(task: Task, transform, config) -> Tuple[TransformerSequenceTokenizer, TransformList]:
    encoder_transform = task.build_tokenizer(transform)
    length_transform = FieldLength('token', 'token_length')
    transform = TransformList(encoder_transform, length_transform)
    extra_transform = config["transform"]
    if extra_transform:
        module_name, class_name = extra_transform["classpath"].rsplit('.', 1)
        normalize = getattr(importlib.import_module(module_name), class_name)
        transform.insert(0, normalize(extra_transform["mapper"], extra_transform["src"]))
    return encoder_transform, transform

class ContextualWordEmbeddingModule(TransformerEncoder):
    def __init__(self,
                 field: str,
                 transformer: str,
                 transformer_tokenizer: PreTrainedTokenizer,
                 average_subwords=False,
                 scalar_mix: Union[ScalarMixWithDropoutBuilder, int] = None,
                 word_dropout=None,
                 max_sequence_length=None,
                 ret_raw_hidden_states=False,
                 transformer_args: Dict[str, Any] = None,
                 trainable=True,
                 training=True) -> None:
        super().__init__(transformer, transformer_tokenizer, average_subwords, scalar_mix, word_dropout,
                         max_sequence_length, ret_raw_hidden_states, transformer_args, trainable,
                         training)
        self.field = field
        
    def forward(self, batch: dict, mask=None, **kwargs):
        input_ids: torch.LongTensor = batch[f'{self.field}_input_ids']
        token_span: torch.LongTensor = batch.get(f'{self.field}_token_span', None)
        output: Union[torch.Tensor, List[torch.Tensor]] = super().forward(input_ids, token_span=token_span, **kwargs)
        return output

    def get_output_dim(self):
        return self.transformer.config.hidden_size

    def get_device(self):
        device: torch.device = next(self.parameters()).device
        return device

class MultiTaskModel(torch.nn.Module):
    def __init__(self,
                 encoder: torch.nn.Module,
                 scalar_mixes: torch.nn.ModuleDict,
                 decoders: torch.nn.ModuleDict,
                 use_raw_hidden_states: dict,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_raw_hidden_states = use_raw_hidden_states
        self.encoder: ContextualWordEmbeddingModule = encoder
        self.scalar_mixes = scalar_mixes
        self.decoders = decoders

def process_sentence(data, task_name=None, device=device):
    if task_name is None:
        task_names = tasks.keys()
    else:
        task_names = task_name

    output_dict = None
    cls_is_bos=True
    sep_is_eos=True
    results = defaultdict(list)
    for task_name in task_names:
        task = tasks[task_name]
        encoder_transform, transform_tok = build_transform(tasks[task_name], transform, task_config)
        samples = tasks[task_name].build_samples(data, cls_is_bos=True, sep_is_eos=True)
        dataloader = tasks[task_name].build_dataloader(samples, transform=transform_tok, device=device)

        for batch in dataloader:
            if output_dict:
                hidden, raw_hidden = output_dict['hidden'], output_dict['raw_hidden']
            else:
                hidden = model.encoder(batch)
                if isinstance(hidden, tuple):
                    hidden, raw_hidden = hidden
                else:
                    raw_hidden = None
                output_dict = {'hidden': hidden, 'raw_hidden': raw_hidden}
            hidden_states = raw_hidden if model.use_raw_hidden_states[task_name] else hidden
            if task_name in model.scalar_mixes:
                scalar_mix = model.scalar_mixes[task_name]
                h = scalar_mix(hidden_states)
            else:
                if model.scalar_mixes:
                    hidden_states = hidden_states[-1, :, :, :]
                h = hidden_states
            task = tasks[task_name]
            if cls_is_bos and not task.cls_is_bos:
                h = h[:, 1:, :]
            if sep_is_eos and not task.sep_is_eos:
                h = h[:, :-1, :]
            
            batch = task.transform_batch(batch, results=results, cls_is_bos=cls_is_bos, sep_is_eos=sep_is_eos)

            batch['mask'] = mask = lengths_to_mask(batch['token_length'])
            output_dict[task_name] = {
                'output': task.feed_batch(h,
                                    batch=batch,
                                    mask=mask,
                                    decoder=model.decoders[task_name]),
                'mask': mask
            }
    return output_dict

filename = "./model/mtl/model.pt"
tokenizer_path = "./tokenizer/electra-base"
# config_path = "./tokenizer/electra-small/config.json" # for electra-small
# config_path = "./tokenizer/ernie-gram/config.json" # for ernie-gram
config_path = "./tokenizer/electra-base/config.json" # for electra-base
task_config_path = "./model/mtl/config.json"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
config = AutoConfig.from_pretrained(config_path)
pre_model = AutoModel.from_config(config=config)
encoder = ContextualWordEmbeddingModule("token", transformer=pre_model, transformer_tokenizer=tokenizer, word_dropout=.1)

with open(task_config_path, 'r', encoding='utf-8') as f:
    task_config = json.load(f)
tasks = {'con': CRFConstituencyParsing(), 
         'pos/ctb': TransformerTagging(), 
        'tok/fine': TaggingTokenization(),
        'dep': BiaffineDependencyParsing(), 
        'ner/msra': TaggingNamedEntityRecognition(), 
        # 'ner/ontonotes': TaggingNamedEntityRecognition(), 
        # 'ner/pku': TaggingNamedEntityRecognition(), 
        # 'sdp': BiaffineSemanticDependencyParsing(), 
        # 'srl': SpanBIOSemanticRoleLabeling(), 
        # 'tok/coarse': TaggingTokenization(), 
        # 'pos/863': TransformerTagging(), 
        # 'pos/pku': TransformerTagging(), 
        }

encoder_size = encoder.get_output_dim()
scalar_mixes = torch.nn.ModuleDict()
decoders = torch.nn.ModuleDict()
training=False
use_raw_hidden_states = dict()
for task_name, task in tasks.items():
    config = task_config.get(task_name)
    decoder = task.build_model(encoder_size=encoder_size, training=training, **config)
    decoders[task_name] = decoder
    if config["scalar_mix"]:
        scalar_mix = task.scalar_mix.build()
        scalar_mixes[task_name] = scalar_mix
        encoder.scalar_mix = 0
    use_raw_hidden_states[task_name] = task.use_raw_hidden_states
encoder.ret_raw_hidden_states = any(use_raw_hidden_states.values())
model = MultiTaskModel(encoder, scalar_mixes, decoders, use_raw_hidden_states)
model.load_state_dict(torch.load(filename, map_location=device, weights_only=True), strict=False)
model.eval()
model.to(device)

transform = TransformerSequenceTokenizer(tokenizer_path,
                                        'token',
                                        ret_subtokens=False,
                                        ret_subtokens_group=False,
                                        ret_token_span=True,
                                        use_fast=True,)

sentences = [
    ["HanLP为生产环境带来次世代最先进的多语种NLP技术。"],
    ["我的希望是希望张晚霞的背影被晚霞映红。"]
  ]

results = process_sentence(sentences,)
# results = process_sentence(sentences, task_name=["tok/fine", "dep"])
# print(results)
