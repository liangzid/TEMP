import torch
import random
import dataclasses
from .utils import format_belief


class NegativeSamplingDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, inner, transform=None,num_bs_negative=4):
        ## inner --> datasets
        self.inner = inner
        self.transform = transform
        assert hasattr(self.inner, 'ontology')
        assert self.inner.ontology is not None
        self.ontology = {k: sorted(v) for k, v in self.inner.ontology.items()}
        self.num_bs_negative=num_bs_negative
    def __len__(self):
        return 2 * len(self.inner)

    def __getitem__(self, i):
        # item = self.inner[i // 2]
        item = self.inner[i]
        
        return self.transform(item)

        # negtive_bs_list=[]
        # # random sampling for belief state.
        # for x in range(self.num_bs_negative):
        #     negative_sample=random.randrange(len(self.inner))
        #     neg_sample=self.inner[negative_sample]
        #     negtive_bs_list.append(format_belief(neg_sample.raw_belief))
        
        # negative = i % 2
        # if negative:
        #     negative = False
        #     belief, response, context = item.belief, item.response, item.context
        #     raw_belief = item.raw_belief
        #     negative_type = random.randrange(1, 4)
        #     use_new_belief = (negative_type // 2) % 2
        #     use_new_response = negative_type % 2

        #     # Negative resonse
        #     negative_sample = random.randrange(len(self.inner))
        #     neg_sample = self.inner[negative_sample]

        #     if use_new_belief:
        #         raw_belief = neg_sample.raw_belief
        #     if use_new_response:
        #         response = neg_sample.response
        #     belief = format_belief(raw_belief)
        #     item = dataclasses.replace(item, context=context,
        #                                belief=belief,
        #                                raw_belief=raw_belief,
        #                                response=response,
        #                                positive=False)

        # item = dataclasses.replace(item,
                                   # negative_bs_list=negtive_bs_list)


class NegativeSamplerWrapper(torch.utils.data.Sampler):
    def __init__(self, inner):
        self.inner = inner

    @property
    def num_samples(self):
        # return 2 * self.inner.num_samples
        return 1 * self.inner.num_samples

    def __iter__(self):
        for index in iter(self.inner):
            # yield 2 * index
            # yield 2 * index + 1
            yield 1 * index
            # yield 1 * index + 1

    def set_epoch(self, epoch):
        if hasattr(self.inner, 'set_epoch'):
            self.inner.set_epoch(epoch)

    def __len__(self):
        return self.num_samples
