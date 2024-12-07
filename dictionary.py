from trie_file import *
from abc import ABC, abstractmethod
from configurable import *
from typing import List, Tuple, Union, Optional, Dict, Any, TextIO, Sequence, Iterable


class DictInterface(ABC):
    @abstractmethod
    def tokenize(self, text: Union[str, Sequence[str]]) -> List[Tuple[int, int, Any]]:
        pass
    def split(self, text: Union[str, Sequence[str]]) -> List[Tuple[int, int, Any]]:
        offset = 0
        spans = []
        for begin, end, label in self.tokenize(text):
            if begin > offset:
                spans.append(text[offset:begin])
            spans.append((begin, end, label))
            offset = end
        if offset < len(text):
            spans.append(text[offset:])
        return spans

class TrieDict(Trie, DictInterface, Configurable):
    def __init__(self, dictionary: Optional[Union[Dict[Iterable[str], Any], Iterable[str]]] = None) -> None:
        super().__init__(dictionary)

    def tokenize(self, text: Union[str, Sequence[str]]) -> List[Tuple[int, int, Any]]:
        return self.parse_longest(text)

    def split_batch(self, data: List[str]) -> Tuple[List[str], List[int], List[List[Tuple[int, int, Any]]]]:
        new_data, new_data_belongs, parts = [], [], []
        for idx, sent in enumerate(data):
            parts.append([])
            found = self.tokenize(sent)
            if found:
                pre_start = 0
                for start, end, info in found:
                    if start > pre_start:
                        new_data.append(sent[pre_start:start])
                        new_data_belongs.append(idx)
                    pre_start = end
                    parts[idx].append((start, end, info))
                if pre_start != len(sent):
                    new_data.append(sent[pre_start:])
                    new_data_belongs.append(idx)
            else:
                new_data.append(sent)
                new_data_belongs.append(idx)
        return new_data, new_data_belongs, parts

    @staticmethod
    def merge_batch(data, new_outputs, new_data_belongs, parts):
        outputs = []
        segments = []
        for idx in range(len(data)):
            segments.append([])
        for o, b in zip(new_outputs, new_data_belongs):
            dst = segments[b]
            dst.append(o)
        for s, p, sent in zip(segments, parts, data):
            s: list = s
            if p:
                dst = []
                offset = 0
                for start, end, info in p:
                    while offset < start:
                        head = s.pop(0)
                        offset += sum(len(token) for token in head)
                        dst += head
                    if isinstance(info, list):
                        dst += info
                    elif isinstance(info, str):
                        dst.append(info)
                    else:
                        dst.append(sent[start:end])
                    offset = end
                if s:
                    assert len(s) == 1
                    dst += s[0]
                outputs.append(dst)
            else:
                outputs.append(s[0])
        return outputs

    @property
    def config(self):
        return {
            'classpath': classpath_of(self),
            'dictionary': dict(self.items())
        }