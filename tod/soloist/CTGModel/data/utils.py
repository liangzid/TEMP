import re
import random
import copy
import logging
from typing import Callable, Union, Set, Optional, List, Dict, Any, Tuple, MutableMapping  # noqa: 401
import dataclasses
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
import numpy as np

from metric import isPolluted

from queue import Queue

## version 3.0.2 cannot use attention mask for attention_matrix.
## I need to modify the source code of transformers.
## liangzi at 2022.02.07

try:
    from tqdm import trange
except(Exception):
    # We do not require tqdm package to be present
    def trange(x, *args, **kwargs):
        return range(x)

logger = logging.getLogger('data')


@dataclass
class DialogDatasetItem:
    context: Union[List[str], str]
    belief: Union[Dict[str, Dict[str, str]], str] = None
    database: Union[List[Tuple[str, int]], List[Tuple[str, int, Any]], None, str] = None
    response: str = None
    template: str = None # added by liangzi.
    negative_bs_list: Any = None # added by liangzi.
    positive: bool = True
    raw_belief: Any = None
    raw_response: str = None
    # dialogue_act: List[List[str]] = None
    dialogue_act: Any = None
    raw_dialogue_act: Any = None
    bp_weight: float = None

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if name == 'belief' and val is None and self.raw_belief is not None:
            val = format_belief(self.raw_belief)
            self.belief = val

        elif name == "dialogue_act" and val is None and self.raw_dialogue_act is not None:
            val=format_dialogue_act(self.raw_dialogue_act)
            self.dialogue_act=val
            
        return val


@dataclass
class DataCollatorWithPadding:
    tokenizer: Union[transformers.PreTrainedTokenizer,
                     transformers.PreTrainedTokenizerFast]
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor,float]]]) -> Dict[str, torch.Tensor]:
        batch = {
            # "attention_mask":torch.tensor([x["attention_mask"].numpy() for x in features]),
            # 'consistency_labels': torch.tensor([x['consistency_labels'] for x in features], dtype=torch.float32),
            # 'consistency_token_ids': torch.tensor([x['consistency_token_ids'] for x in features], dtype=torch.int64),
            'input_ids': pad_sequence([torch.tensor(x['input_ids'], dtype=torch.int64) for x in features],
                                      batch_first=True, padding_value=self.tokenizer.pad_token_id),
            # 'belief_labels': pad_sequence([torch.tensor(x['belief_labels'], dtype=torch.int64) for x in features],
                                          # batch_first=True, padding_value=-100),
        }
        # if "bp_weight" in features[0].keys():
        #     batch["bp_weight"]=torch.tensor([x["bp_weight"] for x in features])
        # else:
        #     batch["bp_weight"]=torch.tensor([1. for x in features])

        # batch["input_ids"]= pad_sequence([torch.tensor(x['input_ids'], dtype=torch.int64) for x in features],
        #                               batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # print(f"input_ids shape: {batch['input_ids'].shape} ")
        # bs,seqlen=batch["input_ids"].shape

        # if "attention_mask" in features[0]:
        #     batch["attention_mask"]= torch.tensor([x['attention_mask'].numpy() for x in features])
        #     # print(batch["attention_mask"].shape)
        #     batch["attention_mask"]=batch["attention_mask"][:,:seqlen,:seqlen]
        # else:
        #     batch["attention_mask"]=None

        if "response_labels" in features[0]:
            batch["response_labels"]= pad_sequence([torch.tensor(x['response_labels'], dtype=torch.int64) for x in features],batch_first=True, padding_value=-100)
        # else:
        #     batch["response_labels"]=None 

        # if "back_predict_labels" in features[0]:
        #     batch["back_predict_labels"]= pad_sequence([torch.tensor(x['back_predict_labels'], dtype=torch.int64) for x in features],batch_first=True, padding_value=-100)
        # # else:
        # #     batch["back_predict_labels"]=None 

        # if "response_end" in features[0]:
        #     batch["response_end"]=torch.tensor([x['response_end'] for x in features])
            
            # batch["response_end"]=pad_sequence([torch.tensor(x['response_end'],dtype=torch.int64) for x in features],batch_first=True,padding_value=-100)

        # if "negative_masked_labels" in features[0]:
        #     newlist=[]
        #     for x in features:
        #         tensor_negative=[torch.tensor(y,dtype=torch.int64)
        #                          for y in x["negative_masked_labels"]]
        #         # print("tensor_neg: {}".format(tensor_negative))
        #         pad_tensor=pad_sequence(tensor_negative,batch_first=True,padding_value=-100).T
        #         # print("shape of neg: {}".format(pad_tensor.shape))
        #         newlist.append(pad_tensor)
        #     batch["negative_masked_labels"]=\
        #         pad_sequence(newlist,batch_first=True,padding_value=-100).transpose(1,2)
        #     # print("shape of negative_masked_labels: {}".format(batch["negative_masked_labels"].shape))

        return batch

class TokenizerBackInferenceTransformation:
    def __init__(self, tokenizer: transformers.GPT2Tokenizer, max_context_length: int = 500, is_bi=False,bp_weight=0.):
        self.bob, self.eob, self.eokb = tokenizer.convert_tokens_to_ids(
            ['=>', '<|eob|>', '<|eokb|>'])

        self.pd,self.pb,self.pc=tokenizer.convert_tokens_to_ids(["<|pd|>", "<|pb|>", "<|pc|>"])

        self.eos = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length

    def splitHistory(self,history):
        if "User : " in history:
            results=history.split("User : ")
            return results[-1]
        else:
            return history

    def backInferenceFormat(self, data):
        history, belief, database = data.context, data.belief, data.database
        response, positive = data.response, data.positive

        ### format blew is back-inference format.
        inp=[]
        labels=[]
        response_end=0

        if positive:
            ## response
            if response is not None:
                response=self.tokenizer.encode(response) + [self.pd]
                inp+=response
                labels += [-100 for _ in response]

            response_end = len(labels)

            ## database
            if database is not None:
                database = self.tokenizer.encode(database) + [self.pb]
                inp += database
                labels += [-100 for _ in database]

            database_end = len(labels)

            # Add belief states
            if belief is not None:
                belief = self.tokenizer.encode(belief) + [self.pc]
                inp += belief
                labels += belief

            belief_end = len(labels)

            # Add history finally
            history = self.tokenizer.encode(history)
            inp += history
            labels += [-100 for _ in history]
            context_end = len(labels)
        else:
            ## database
            if database is not None:
                database = self.tokenizer.encode(database) + [self.pb]
                inp += database
                labels += [-100 for _ in database]

            database_end = len(labels)

            # Add belief states
            if belief is not None:
                belief = self.tokenizer.encode(belief) + [self.pc]
                inp += belief
                labels += belief

            belief_end = len(labels)

            # Add history finally
            history = self.tokenizer.encode(history)+[self.eos]
            inp += history
            labels += [-100 for _ in history]
            context_end = len(labels)

        if self.max_context_length > 0:

            old_length = len(inp)
            inp = inp[:(self.max_context_length-1)]+[self.eos]
            labels = labels[:(self.max_context_length-1)]+[self.eos]
            assert(len(inp)==len(labels))

            # belief_end = belief_end - (old_length - len(inp))
            # context_end = context_end - (old_length - len(inp))
            # database_end = database_end - (old_length - len(inp))

        return inp, labels, positive, response_end, database_end, belief_end

    # -100 is mask token for LM
    # transforms into dict {"input_ids", "labels", "binary_labels", "binary_token_ids" }
    # binary_labels are used for task 3
    def __call__(self, data):
        inp, labels, positive, response_end, database_end, belief_end = self.backInferenceFormat(data)
        belief_labels = [x if i < belief_end else -100 for i, x in enumerate(labels)]
        context_labels = [x if i >= belief_end else -100 for i, x in enumerate(labels)]

        return dict(input_ids=inp, belief_labels=belief_labels, context=context_labels,
                    consistency_labels=positive, consistency_token_ids=len(labels) - 1)


class TokenizerBidirectionalGenerationTransformation:
    def __init__(self, tokenizer: transformers.GPT2Tokenizer, max_context_length: int = 500,
                 is_bi=False, make_mask=0,bp_weight=0.):

        self.make_mask=make_mask
        self.bob, self.eob, self.eokb = tokenizer.convert_tokens_to_ids(
            ['=>', '<|eob|>', '<|eokb|>'])

        self.pd,self.pb,self.pc=tokenizer.convert_tokens_to_ids(["<|pd|>", "<|pb|>", "<|pc|>"])

        self.eos = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.negative_masked_labels=Queue(4);

    def splitHistory(self,history):
        if "User : " in history:
            results=history.split("User : ")
            return results[-1]
        else:
            return history

    def backInferenceFormat(self, data):
        history, belief, database = data.context, data.belief, data.database
        response, positive = data.response, data.positive

        ### format blew is back-inference format.
        inp=[]
        labels=[]
        response_end=0

        # Add history
        history = self.tokenizer.encode(history)
        inp = history
        labels = [-100 for _ in history]
        context_end = len(labels)

        # Add belief states
        if belief is not None:
            belief1 = [self.bob] + self.tokenizer.encode(belief) + [self.eob]
            inp += belief1
            labels += belief1

        belief_end = len(labels)

        # Add database
        if database is not None:
            database1 = self.tokenizer.encode(database) + [self.eokb]
            inp += database1
            labels += [-100 for _ in database1]

        database_end = len(labels)

        # Add response
        if response is not None:
            response = self.tokenizer.encode(response) + [self.pb]
            inp += response
            labels += response

        bresponse_end=len(labels)

        # ## database
        # if database is not None:
        #     database = self.tokenizer.encode(database) + [self.pb]
        #     inp += database
        #     # labels += [-100 for _ in database]
        #     labels += database 

        # database_end = len(labels)

        # Add belief states
        if belief is not None:
            belief1 = self.tokenizer.encode(belief) + [self.pd]
            inp += belief1
            labels += belief1
        # Add database
        if database is not None:
            database1 = self.tokenizer.encode(database) + [self.eos] 
            inp += database1
            # labels += [-100 for _ in database1]
            labels += database1 

        bbelief_end = len(labels)


        # # Add history finally
        # history = self.tokenizer.encode(history) + [self.eos]
        # inp += history
        # # labels += [-100 for _ in history]
        # labels += history 
        # # bcontext_end = len(labels)


        if positive is not None and not positive:
            labels = [-100 for _ in labels]

        if self.max_context_length > 0:

            old_length = len(inp)
            inp = inp[-self.max_context_length:]
            labels = labels[-self.max_context_length:]

            belief_end = belief_end - (old_length - len(inp))
            context_end = context_end - (old_length - len(inp))
            database_end = database_end - (old_length - len(inp))

            bresponse_end = bresponse_end - (old_length - len(inp))
            bbelief_end = bbelief_end - (old_length - len(inp))

        # print(labels[bresponse_end:].__len__())
        if positive is None or positive:
            if self.negative_masked_labels.full():
                self.negative_masked_labels.get()
            self.negative_masked_labels.put(labels[bresponse_end:])
        else:
            pass
            
        # if self.max_context_length > 0:

        #     old_length = len(inp)
        #     inp = inp[:(self.max_context_length-1)]+[self.eos]
        #     labels = labels[:(self.max_context_length-1)]+[self.eos]
        #     assert(len(inp)==len(labels))

        #     # belief_end = belief_end - (old_length - len(inp))
        #     # context_end = context_end - (old_length - len(inp))
        #     # database_end = database_end - (old_length - len(inp))

        # print(f"tokens: {inp}")
        # print(f"labels: {labels}")

        if self.make_mask==1:
            sequence_length=len(inp)
            sequence_length=self.max_context_length
            attention_mask=torch.ones((sequence_length,sequence_length))
            # attention_mask[bresponse_end:,:belief_end]=0
            attention_mask[:belief_end,bresponse_end:]=0
        else:
            attention_mask=None

        return inp, labels, positive, belief_end, context_end,database_end, bresponse_end, bbelief_end,attention_mask

    # -100 is mask token for LM
    # transforms into dict {"input_ids", "labels", "binary_labels", "binary_token_ids" }
    # binary_labels are used for task 3
    def __call__(self, data):
        inp, labels, positive, belief_end, context_end, database_end, bresponse_end, bbelief_end,attention_mask = self.backInferenceFormat(data)
        belief_labels = [x if i < belief_end else -100 for i, x in enumerate(labels)]
        response_labels = [x if i >= belief_end and x<bresponse_end else -100 for i, x in enumerate(labels)]
        back_predict_labels = [x if i >= bresponse_end else -100 for i, x in enumerate(labels)]

        # return dict(input_ids=inp, belief_labels=belief_labels, response_labels=context_labels,
        #             consistency_labels=positive, consistency_token_ids=len(labels) - 1)

        return dict(input_ids=inp,
                    belief_labels=belief_labels,
                    response_labels=response_labels,
                    response_end=bresponse_end,
                    negative_masked_labels=list(self.negative_masked_labels.queue),
                    back_predict_labels=back_predict_labels,
                    consistency_labels=positive,
                    consistency_token_ids=len(labels) - 1,
                    attention_mask=attention_mask)


class TokenizerTransformation:
    def __init__(self, tokenizer: transformers.GPT2Tokenizer, max_context_length: int = 500, is_bi=False):
        self.bob, self.eob, self.eokb = tokenizer.convert_tokens_to_ids(
            ['=>', '<|eob|>', '<|eokb|>'])

        # self.pd,self.pb,self.pc=tokenizer.convert_tokens_to_ids(["<|pd|>", "<|pb|>", "<|pc|>"])

        self.eos = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length

    def get_tokens(self, data):
        history, belief, database = data.context, data.belief, data.database
        response, positive = data.response, data.positive

        # Add history
        history = self.tokenizer.encode(history)
        inp = history
        labels = [-100 for _ in history]
        context_end = len(labels)

        # Add belief states
        if belief is not None:
            belief = [self.bob] + self.tokenizer.encode(belief) + [self.eob]
            inp += belief
            labels += belief

        belief_end = len(labels)

        # Add database
        if database is not None:
            database = self.tokenizer.encode(database) + [self.eokb]
            inp += database
            labels += [-100 for _ in database]

        database_end = len(labels)

        # Add response
        if response is not None:
            response = self.tokenizer.encode(response) + [self.eos]
            inp += response
            labels += response

        if positive is not None and not positive:
            labels = [-100 for _ in labels]

        if self.max_context_length > 0:

            old_length = len(inp)
            inp = inp[-self.max_context_length:]
            labels = labels[-self.max_context_length:]

            belief_end = belief_end - (old_length - len(inp))
            context_end = context_end - (old_length - len(inp))
            database_end = database_end - (old_length - len(inp))

        return inp, labels, positive, belief_end, context_end, database_end

    # -100 is mask token for LM
    # transforms into dict {"input_ids", "labels", "binary_labels", "binary_token_ids" }
    # binary_labels are used for task 3
    def __call__(self, data):
        inp, labels, positive, belief_end, context_end, database_end = self.get_tokens(data)
        belief_labels = [x if i < belief_end else -100 for i, x in enumerate(labels)]
        response_labels = [x if i >= belief_end else -100 for i, x in enumerate(labels)]
        return dict(input_ids=inp, belief_labels=belief_labels, response_labels=response_labels,
                    consistency_labels=positive, consistency_token_ids=len(labels) - 1)


def default_translate_match(n):
    if n == 0:
        return 'no match'
    if n == 1:
        return '1 match'
    return f'{n} matches'


@dataclass
class InsertLabelsTransformation:
    user_label: str = 'User :'
    sys_label: str = 'System :'
    database_label: str = 'DB :'
    belief_label: str = 'Belief state :'
    dialogue_act_label: str = 'Action :'
    # template_label: str = "Template :"

    def __call__(self, sample: DialogDatasetItem) -> DialogDatasetItem:
        if isinstance(sample, tuple):
            sample = DialogDatasetItem(*sample)
        # # Transform context
        # context = sample.context
        # context = list(context)
        # labels = self.user_label, self.sys_label
        # for i in range(len(context) - 1, -1, -1):
        #     label, other = labels
        #     context[i] = label + ' ' + context[i]
        #     labels = other, label
        # context = ' '.join(context)

        # # Database
        # database = sample.database
        # if database is not None:
        #     database_str = []
        #     for database_domain, database_count in database.items():
        #         database_str.append(database_domain + ' ' +
        #                             default_translate_match(database_count))
        #     database = self.database_label + ' ' + ' , '.join(database_str)

        # # Belief state
        # belief = sample.belief
        # if belief is not None:
        #     belief = self.belief_label + ' ' + belief

        # dialogue act
        if sample.dialogue_act is not None:
            dialogue_act=sample.dialogue_act
            dialogue_act=self.dialogue_act_label+" "+dialogue_act
        else:
            dialogue_act=None

        # # template
        # template=sample.template
        # if template is not None:
        #     template=self.template_label+" "+ str()
        # return dataclasses.replace(sample, belief=belief,
                                   # database=database, context=context,template=template)

        return dataclasses.replace(sample, 
                                   dialogue_act=dialogue_act,
                                   )


class BeliefParser:
    def __init__(self):
        self.slotval_re = re.compile(r"(\w[\w ]*\w) = ([\w\d: |']+)")
        self.domain_re = re.compile(r"(\w+) {\s*([\w,= :\d|']*)\s*}", re.IGNORECASE)

    def __call__(self, raw_belief: str):
        belief = OrderedDict()
        for match in self.domain_re.finditer(raw_belief):
            domain, domain_bs = match.group(1), match.group(2)
            belief[domain] = {}
            for slot_match in self.slotval_re.finditer(domain_bs):
                slot, val = slot_match.group(1), slot_match.group(2)
                belief[domain][slot] = val
        return belief


def format_belief(belief: OrderedDict) -> str:
    assert isinstance(belief, OrderedDict)
    str_bs = []
    for domain, domain_bs in belief.items():
        domain_bs = ', '.join([f'{slot} = {val}' for slot, val in sorted(domain_bs.items(), key=lambda x: x[0])])
        str_bs.extend([domain, '{' + domain_bs + '}'])
    return ' '.join(str_bs)

## not done.
def format_dialogue_act(acts) -> str:
    # assert isinstance(acts, List[List[str]])
    str_da = []
    # intent-domain-slot-value
    str_acts=""
    for act in acts:
        intent,domain,slot,value=act
        str_acts+=f"{intent}, {domain}, {slot}, {value}; "
    return str_acts[:-2]

class FakeDatabase:
    def __init__(self, seed=None):
        self._rnd = random.Random(seed)

    def __call__(self, belief, return_results=False) \
            -> "Union[OrderedDict[str, Tuple[int, dict]], OrderedDict[str, int]]":
        results = OrderedDict()
        for key, bs in belief.items():
            count = random.randrange(-5, 15)
            items = [{} for i in range(count)]
            results[key] = (len(items), items) if return_results else len(items)
        return results


def merge_ontologies(ontologies):
    ontology = defaultdict(lambda: set())
    for o in ontologies:
        if o is None:
            continue
        for k, val in o.items():
            ontology[k].update(val)
    return ontology
    
@dataclass
class DialogDataset(torch.utils.data.Dataset):
    items: List[any]
    database: Any = None
    domains: List[str] = None
    lexicalizer: Any = None
    transform: Callable[[Any], Any] = None
    normalize_input: Callable[[str], str] = None
    ontology: Dict[Tuple[str, str], Set[str]] = None

    @staticmethod
    def build_dataset_without_database(items, *args, **kwargs):
        return DialogDataset(items, FakeDatabase(), *args, **kwargs)

    def __getitem__(self, index):
        item = self.items[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.items)

    def map(self, transformation):
        def trans(x):
            x = self.transform(x)
            x = transformation(x)
            return x
        return dataclasses.replace(self, transform=trans)

    def finish(self, progressbar: Union[str, bool] = False):
        if self.transform is None:
            return self

        ontology = defaultdict(lambda: set())
        domains = set(self.domains) if self.domains else set()

        items = []
        for i in trange(len(self),
                        desc=progressbar if isinstance(progressbar, str) else 'loading dataset',
                        disable=not progressbar):
            item = self[i]
            for k, bs in item.raw_belief.items():
                domains.add(k)
                for k2, val in bs.items():
                    ontology[(k, k2)].add(val)
            items.append(item)
        if self.ontology:
            ontology = merge_ontologies((self.ontology, ontology))
        return dataclasses.replace(self, items=items, transform=None, domains=domains, ontology=ontology)


def wrap_dataset_with_cache(dataset):
    dataset = copy.copy(dataset)
    old_get = dataset.__getitem__
    cache = dict()

    def cached_get(i):
        if i not in cache:
            cache[i] = old_get(i)
        return cache[i]
    dataset.__getitem__ = cached_get
    return dataset


class ConcatDialogDataset(torch.utils.data.ConcatDataset):
    def map(self, transformation):
        return ConcatDialogDataset([x.map(transformation) for x in self.datasets])

    def finish(self, progressbar: Union[str, bool] = False):
        dataset = DialogDataset(self, database=FakeDatabase(), transform=lambda x: x)
        return dataset.finish(progressbar)


class BlacklistItemsWrapper:
    def __init__(self, items, blacklist):
        self.items = items
        self._indexmap = []

        blacklist_pointer = 0
        for i in range(len(items)):
            if blacklist_pointer >= len(blacklist):
                self._indexmap.append(i)
            elif i < blacklist[blacklist_pointer]:
                self._indexmap.append(i)
            elif i == blacklist[blacklist_pointer]:
                blacklist_pointer += 1

        assert len(self._indexmap) == len(items) - len(blacklist)

    def __getitem__(self, idx):
        return self.items[self._indexmap[idx]]

    def __len__(self):
        return len(self._indexmap)


def wrap_dataset_with_blacklist(dataset, blacklist):
    return dataclasses.replace(dataset, items=BlacklistItemsWrapper(dataset.items, blacklist))


def split_name(dataset_name: str):
    split = dataset_name.rindex('-')
    return dataset_name[:split], dataset_name[split + 1:]


def sort_database(belief: OrderedDict, database: Dict[str, Dict[str, str]]) -> OrderedDict:
    database = {k: v for k, v in database.items()}
    first_db = None
    if belief:
        first_key = next(iter(belief.keys()))
        first_db = database.pop(first_key, None)
    items = [(first_key, first_db)] if first_db is not None else []
    items += [(k, v) for k, v in sorted(database.items(), key=lambda x: x[0])]
    return OrderedDict(items)


def sort_belief(belief: dict, active_domain: Optional[str]):
    belief = {k: OrderedDict(sorted(v.items(), key=lambda x: x[0])) for k, v in belief.items()}
    if active_domain is not None:
        active_domain = active_domain.lower()
        active_bs = belief.pop(active_domain, None)
    else:
        active_bs = None

    items = [(active_domain, active_bs)] if active_bs is not None else []
    items += [(k, v) for k, v in sorted(belief.items(), key=lambda x: x[0])]
    result = OrderedDict(items)
    return result


class ScgptTransformation:
    def __init__(self, tokenizer: transformers.GPT2Tokenizer, max_context_length: int = 500,
                 is_bi=False, make_mask=0,bp_weight=0.):

        self.make_mask=make_mask
        self.bob, self.eob, self.eokb = tokenizer.convert_tokens_to_ids(
            ['=>', '<|eob|>', '<|eokb|>'])

        # self.pd,self.pb,self.pc=tokenizer.convert_tokens_to_ids(["<|pd|>", "<|pb|>", "<|pc|>"])
        # self.pd,self.pb,self.pc=tokenizer.convert_tokens_to_ids(["<|pd|>", "<|pb|>", "<|pc|>"])
        self.pa,self.eoda=tokenizer.convert_tokens_to_ids(["<|pa|>","<|eoda|>"])

        self.eos = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.get_tokens=self.backInferenceFormat

        # self.negative_masked_labels=Queue(4);

    def splitHistory(self,history):
        if "User : " in history:
            results=history.split("User : ")
            return results[-1]
        else:
            return history

    def backInferenceFormat(self, data):
        # history, belief, database = data.context, data.belief, data.database
        response, positive = data.response, data.positive
        dialogue_act=data.dialogue_act

        ### format blew is back-inference format.
        inp=[]
        labels=[]
        response_end=0

        # # Add history
        # history = self.tokenizer.encode(history)
        # inp = history
        # labels = [-100 for _ in history]
        # context_end = len(labels)

        # # Add belief states
        # if belief is not None:
        #     belief1 = [self.bob] + self.tokenizer.encode(belief) + [self.eob]
        #     inp += belief1
        #     labels += belief1

        # belief_end = len(labels)

        # # Add database
        # if database is not None:
        #     database1 = self.tokenizer.encode(database) + [self.eokb]
        #     inp += database1
        #     labels += [-100 for _ in database1]

        # database_end = len(labels)

        if dialogue_act is not None:
            da1=self.tokenizer.encode(dialogue_act) + [self.eoda]
            inp+=da1
            labels+=[-1 for x in da1]

        # Add response
        if response is not None:
            response = self.tokenizer.encode(response) + [self.eos]
            inp += response
            labels += response

        bresponse_end=len(labels)


        if positive is not None and not positive:
            labels = [-100 for _ in labels]

        if self.max_context_length > 0:

            old_length = len(inp)
            inp = inp[-self.max_context_length:]
            labels = labels[-self.max_context_length:]

        # # print(labels[bresponse_end:].__len__())
        # if positive is None or positive:
        #     if self.negative_masked_labels.full():
        #         self.negative_masked_labels.get()
        #     self.negative_masked_labels.put(labels[bresponse_end:])
        # else:
        #     pass
            
        # if self.max_context_length > 0:

        #     old_length = len(inp)
        #     inp = inp[:(self.max_context_length-1)]+[self.eos]
        #     labels = labels[:(self.max_context_length-1)]+[self.eos]
        #     assert(len(inp)==len(labels))

        #     # belief_end = belief_end - (old_length - len(inp))
        #     # context_end = context_end - (old_length - len(inp))
        #     # database_end = database_end - (old_length - len(inp))

        # print(f"tokens: {inp}")
        # print(f"labels: {labels}")

        # if self.make_mask==1:
        #     sequence_length=len(inp)
        #     sequence_length=self.max_context_length
        #     attention_mask=torch.ones((sequence_length,sequence_length))
        #     # attention_mask[bresponse_end:,:belief_end]=0
        #     attention_mask[:belief_end,bresponse_end:]=0
        # else:
        #     attention_mask=None

        return inp, labels

    # -100 is mask token for LM
    # transforms into dict {"input_ids", "labels", "binary_labels", "binary_token_ids" }
    # binary_labels are used for task 3
    def __call__(self, data):
        inp, labels = self.backInferenceFormat(data)
        response_labels = [x for i, x in enumerate(labels)]

        return dict(input_ids=inp,
                    response_labels=response_labels,
                    )

class CTGScgptTransformation:
    def __init__(self, tokenizer: transformers.GPT2Tokenizer, max_context_length: int = 500,
                 is_bi=False, make_mask=0,bp_weight=0.):

        self.make_mask=make_mask
        self.bob, self.eob, self.eokb = tokenizer.convert_tokens_to_ids(
            ['=>', '<|eob|>', '<|eokb|>'])

        # self.pd,self.pb,self.pc=tokenizer.convert_tokens_to_ids(["<|pd|>", "<|pb|>", "<|pc|>"])
        # self.pd,self.pb,self.pc=tokenizer.convert_tokens_to_ids(["<|pd|>", "<|pb|>", "<|pc|>"])
        self.pa,self.eoda=tokenizer.convert_tokens_to_ids(["<|pa|>","<|eoda|>"])
        self.sep=tokenizer.convert_tokens_to_ids(["<|sep|>"])[0]

        self.eos = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.get_tokens=self.backInferenceFormat

        # self.negative_masked_labels=Queue(4);

    def splitHistory(self,history):
        if "User : " in history:
            results=history.split("User : ")
            return results[-1]
        else:
            return history

    def backInferenceFormat(self, data):
        # history, belief, database = data.context, data.belief, data.database
        response, positive = data.response, data.positive
        dialogue_act=data.dialogue_act

        ## here we adding a new metric for descriminate whether the sentence is offensive or not.
        ispolluted=isPolluted(data.response)

        ### format blew is back-inference format.
        inp=[]
        labels=[]
        response_end=0

        # # Add history
        # history = self.tokenizer.encode(history)
        # inp = history
        # labels = [-100 for _ in history]
        # context_end = len(labels)

        # # Add belief states
        # if belief is not None:
        #     belief1 = [self.bob] + self.tokenizer.encode(belief) + [self.eob]
        #     inp += belief1
        #     labels += belief1

        # belief_end = len(labels)

        # # Add database
        # if database is not None:
        #     database1 = self.tokenizer.encode(database) + [self.eokb]
        #     inp += database1
        #     labels += [-100 for _ in database1]

        # database_end = len(labels)

        if positive and response is not None:
            if ispolluted-1.0==0.0: # means this response is polluted.
                words=self.tokenizer.encode("Offensive yes")+[self.sep]
            else:
                words=self.tokenizer.encode("Offensive no")+[self.sep]
            inp+=words
            labels+=[-100 for x in words]

        ## this means at inference time. We only input no offensive attribute.
        if positive and response is None:
            words=self.tokenizer.encode("Offensive no")+[self.sep]
            inp+=words
            labels+=[-100 for x in words]

        if dialogue_act is not None:
            da1=self.tokenizer.encode(dialogue_act) + [self.eoda]
            inp+=da1
            labels+=[-100 for x in da1]

        # Add response
        if response is not None:
            response = self.tokenizer.encode(response) + [self.eos]
            inp += response
            labels += response

        bresponse_end=len(labels)


        if positive is not None and not positive:
            labels = [-100 for _ in labels]

        if self.max_context_length > 0:

            old_length = len(inp)
            inp = inp[-self.max_context_length:]
            labels = labels[-self.max_context_length:]

        # # print(labels[bresponse_end:].__len__())
        # if positive is None or positive:
        #     if self.negative_masked_labels.full():
        #         self.negative_masked_labels.get()
        #     self.negative_masked_labels.put(labels[bresponse_end:])
        # else:
        #     pass
            
        # if self.max_context_length > 0:

        #     old_length = len(inp)
        #     inp = inp[:(self.max_context_length-1)]+[self.eos]
        #     labels = labels[:(self.max_context_length-1)]+[self.eos]
        #     assert(len(inp)==len(labels))

        #     # belief_end = belief_end - (old_length - len(inp))
        #     # context_end = context_end - (old_length - len(inp))
        #     # database_end = database_end - (old_length - len(inp))

        # print(f"tokens: {inp}")
        # print(f"labels: {labels}")

        # if self.make_mask==1:
        #     sequence_length=len(inp)
        #     sequence_length=self.max_context_length
        #     attention_mask=torch.ones((sequence_length,sequence_length))
        #     # attention_mask[bresponse_end:,:belief_end]=0
        #     attention_mask[:belief_end,bresponse_end:]=0
        # else:
        #     attention_mask=None

        return inp, labels

    # -100 is mask token for LM
    # transforms into dict {"input_ids", "labels", "binary_labels", "binary_token_ids" }
    # binary_labels are used for task 3
    def __call__(self, data):
        inp, labels = self.backInferenceFormat(data)
        response_labels = [x for i, x in enumerate(labels)]

        # print("===========================")
        # print(self.tokenizer.decode(inp))

        return dict(input_ids=inp,
                    response_labels=response_labels,
                    )
