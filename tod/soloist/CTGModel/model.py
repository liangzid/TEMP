import dataclasses
import logging
from dataclasses import dataclass
from transformers import GPT2Tokenizer as SoloistTokenizer  # noqa
from torch import nn
# import transformers
import sys
sys.path.append("/home/liangzi/code/transformers-302/")
import src.transformers as transformers

from torch.nn import functional as F
import torch
import data


EOB_TK = '<|eob|>'
EOKB_TK = '<|eokb|>'
EOT_TK = '<|endoftext|>'

SPECIAL_TOKENS = [EOB_TK, EOKB_TK,
                  "<|pd|>","<|pb|>","<|pc|>",
                  "<|pa|>","<|eoda|>","<|sep|>"]
logger = logging.getLogger()


def add_custom_tokens(tokenizer, model):
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


# TODO: new transformers version
# @dataclass
# class SoloistModelOutput(transformers.ModelOutput):
#     """
#     SoloistModelOutput with consistency detection, split loss between belief state and response
#
#     Args:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
#             Language modeling loss.
#         mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
#             Multiple choice classification loss.
#         logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#         mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
#             Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
#         past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
#             List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
#             :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
#
#             Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
#             :obj:`past_key_values` input) to speed up sequential decoding.
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.
#
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#     """
#
#     loss: Optional[torch.FloatTensor] = None
#     mc_loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None
#     mc_logits: torch.FloatTensor = None
#     past_key_values: Optional[List[torch.FloatTensor]] = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None


class SoloistConfig(transformers.GPT2Config):
    def __init__(self,
                 summary_label_smoothing=0.1, # for overfitting.
                 **kwargs):
        super().__init__(**kwargs)
        self.summary_label_smoothing = summary_label_smoothing


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """please reference at https://blog.csdn.net/weixin_44305115/article/details/106605237"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(-1)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * pred * (target != -100).unsqueeze(-1))
        return loss / (target != -100).sum()


class LabelSmoothingBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target, weight=None):
        smoothed_labels = target.mul(1 - 2 * self.smoothing).add_(self.smoothing)
        return torch.nn.functional.binary_cross_entropy_with_logits(input, smoothed_labels, weight)

class MaskTODSoloistModel(transformers.GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias",
                               r"lm\_head\.weight", r"binary\_head\.\w+"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = transformers.GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.consistency_head = nn.Linear(config.n_embd, 1) # ?
        self.auxiliary_dropout = nn.Dropout(config.summary_first_dropout)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self,
                input_ids=None,          # all input sequence tokens;
                past=None,

                attention_mask=None,
                response_end=None, ## mark the end of responses, for special usages.
                negative_masked_labels=None, ## used to construct negative labels.
                back_predict_labels=None,  # this is the target of back predicted resonses.
                bp_weight=0.3,
                consistency_labels=None,  # is consistency or not

                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                consistency_token_ids=None, # the last token (eos), for classify whether consistent or not. 
                user_intent_token_ids=None,
                user_intent_labels=None,
                user_intent_mask=None,
                belief_labels=None,      # context + belief states, and aims to predict bs.
                system_action_token_ids=None,
                system_action_labels=None,
                system_action_mask=None,
                response_labels=None, # only responses part has label, and others part is -100.
                binary_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                **kwargs  # context=context_labels
                ):
        # print(f"shape of bp weight: {bp_weight.shape}")
        # print("forward.")

        # print(f"{input_ids.shape}")
        # print(f"{attention_mask.shape}")
        transformer_outputs = self.transformer(
            input_ids.contiguous(),
            past=past,
            attention_mask=attention_mask.contiguous(),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # print(transformer_outputs)

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states).contiguous()

        def gather_auxiliary_features(token_ids):
            if token_ids is None:
                # torch.full_like(input,fill_value) returns the same size
                #as input with the filling of fill_value.
                
                # hidden_states.shape[-2] is max-seqence-length

                # ... in split means select the last dimension, so it means select the last
                # embedding dimension, and select the first batch, and with all seqence.
                # so the shape is 1*msl*1
                token_ids = torch.full_like(hidden_states[..., :1, :], # which the shape is ???
                                            hidden_states.shape[-2]-1, dtype=torch.long,)
            else:
                token_ids = token_ids.unsqueeze(-1).unsqueeze(-1)
                token_ids = token_ids.expand(
                    (-1,) * (token_ids.dim() - 1) + (hidden_states.size(-1),))

            # shape of binary_token_ids: (bsz, XX, 1, hidden_size)
            # where XX are optional leading dim of hidden_states
            # shape of binary_logits (bsz, XX, hidden_size)
            logits = hidden_states.gather(-2, token_ids).squeeze(-2)
            logits = self.auxiliary_dropout(logits)
            return logits

        consistency_logits = self.consistency_head(gather_auxiliary_features(consistency_token_ids)).squeeze(-1)
        consistency_loss = None
        if consistency_labels is not None:
            # Auxiliary tasks
            aux_criterion = LabelSmoothingBCEWithLogitsLoss(self.config.summary_label_smoothing)
            consistency_loss = aux_criterion(consistency_logits, consistency_labels)

        ## here we calculate the discriminative loss.
        bp_loss=0.
        if back_predict_labels is not None and consistency_labels is not None and True:
            assert response_end is not None 
            assert negative_masked_labels is not None 
            # print(f"negative masked labels: {negative_masked_labels.shape}")
            # print("shape of attention_mask: ",attention_mask.shape)

            pn_loss_fct=nn.CrossEntropyLoss()

            bs,msl,d=hidden_states.shape
            # print(bs,msl,d)
            # print(negative_masked_labels.shape)
            for b in range(bs):
                if consistency_labels[b]!=1:
                    continue
                else:
                    neg_losses=torch.zeros(negative_masked_labels.shape[1]+1)
                    for i, this_n_label in enumerate(negative_masked_labels[b]):
                        new_input=torch.cat((input_ids[b,:response_end[b]],
                                             this_n_label)).unsqueeze(0)
                        # print("shape of input: ",new_input.shape)
                        if msl<new_input.shape[1]:
                            new_input=new_input[:,:msl]
                        elif msl==new_input.shape[1]:
                            continue
                        else:
                            new_input=torch.cat((new_input,
                                                    torch.ones((1,msl-new_input.shape[1]),
                                                               dtype=torch.int64,device=new_input.get_device())*(-100)),
                                                dim=1)
                        new_input[new_input==-100]=0
                        # print("shape of input: ",new_input.shape)
                        x=self.transformer(new_input,past=past,
                             attention_mask=attention_mask[b].unsqueeze(0).contiguous(),
                                              token_type_ids=token_type_ids,
                                              position_ids=position_ids,
                                              head_mask=head_mask,
                                              inputs_embeds=inputs_embeds,
                                              use_cache=use_cache,
                                              output_attentions=output_attentions,
                                              output_hidden_states=output_hidden_states,
                                              )
                        # print(x.shape)
                        out_logits=self.lm_head(x[0])[...,:-1,:].contiguous()
                        shift_ol=torch.cat((response_labels[b],
                                            this_n_label))[:msl].unsqueeze(0)[...,1:].contiguous()

                        
                        neg_losses[i]=pn_loss_fct(out_logits.view(shift_ol.size(-1),-1),
                                                shift_ol.view(-1))

                    # After calculate negative losses, we now calculate the ground Truth losses.
                    shift_labels=torch.cat((response_labels[b],
                                            back_predict_labels[b]))[:msl].unsqueeze(0)[...,1:].contiguous()
                    # print(lm_logits.shape)
                    # print(shift_labels.shape)
                    posi_loss=pn_loss_fct(lm_logits[b,
                                                    :-1,:].unsqueeze(0).view(shift_labels.size(-1)
                                                                             ,-1),
                                          shift_labels.view(-1))
                    neg_losses[-1]=posi_loss
                    bp_los=nn.Softmax(dim=0)(neg_losses)[-1]
                    bp_loss+=bp_los
                    
        # print(bp_loss)
                    
        belief_loss, response_loss = None, None
        if belief_labels is not None:
            assert response_labels is not None

            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_belief_labels = belief_labels[..., 1:].contiguous()
            shift_response_labels = response_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            belief_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_belief_labels.view(-1))

            response_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_response_labels.view(-1))
            response_loss = response_ce

            # ## we only use 0.5 weighted bp losses.
            # if back_predict_labels is not None:
            #     shift_bp_labels = back_predict_labels[..., 1:].contiguous()
            #     bp_loss = loss_fct(
            #     shift_logits.view(-1, shift_logits.size(-1)),
            #     shift_bp_labels.view(-1))
            #     # assert bp_weight is not None
            #     bp_loss*=bp_weight[0]
        
        # print(consistency_logits)
        output = (lm_logits, consistency_logits,) + transformer_outputs[1:]
        if consistency_loss is not None:
            output = (consistency_loss,) + output
        return ((belief_loss, response_loss, response_ce,bp_loss,) + output) if belief_loss is not None else output


class SoloistModel(transformers.GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias",
                               r"lm\_head\.weight", r"binary\_head\.\w+"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = transformers.GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.consistency_head = nn.Linear(config.n_embd, 1) # ?
        self.auxiliary_dropout = nn.Dropout(config.summary_first_dropout)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self,
                input_ids=None,          # all input sequence tokens;
                past=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                consistency_token_ids=None, # the last token (eos), for classify whether consistent or not. 
                consistency_labels=None,  # is consistency or not
                user_intent_token_ids=None,
                user_intent_labels=None,
                user_intent_mask=None,
                belief_labels=None,      # context + belief states, and aims to predict bs.
                system_action_token_ids=None,
                system_action_labels=None,
                system_action_mask=None,
                response_labels=None, # only responses part has label, and others part is -100.
                back_predict_labels=None,  # this is the target of back predicted resonses.
                bp_weight=0.3,
                binary_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                **kwargs  # context=context_labels
                ):
        # print(f"shape of bp weight: {bp_weight.shape}")

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        def gather_auxiliary_features(token_ids):
            if token_ids is None:
                # torch.full_like(input,fill_value) returns the same size
                #as input with the filling of fill_value.
                
                # hidden_states.shape[-2] is max-seqence-length

                # ... in split means select the last dimension, so it means select the last
                # embedding dimension, and select the first batch, and with all seqence.
                # so the shape is 1*msl*1
                token_ids = torch.full_like(hidden_states[..., :1, :], # which the shape is ???
                                            hidden_states.shape[-2]-1, dtype=torch.long,)
            else:
                token_ids = token_ids.unsqueeze(-1).unsqueeze(-1)
                token_ids = token_ids.expand(
                    (-1,) * (token_ids.dim() - 1) + (hidden_states.size(-1),))

            # shape of binary_token_ids: (bsz, XX, 1, hidden_size)
            # where XX are optional leading dim of hidden_states
            # shape of binary_logits (bsz, XX, hidden_size)
            logits = hidden_states.gather(-2, token_ids).squeeze(-2)
            logits = self.auxiliary_dropout(logits)
            return logits

        consistency_logits = self.consistency_head(gather_auxiliary_features(consistency_token_ids)).squeeze(-1)
        consistency_loss = None
        if consistency_labels is not None:
            # Auxiliary tasks
            aux_criterion = LabelSmoothingBCEWithLogitsLoss(self.config.summary_label_smoothing)
            consistency_loss = aux_criterion(consistency_logits, consistency_labels)

        belief_loss, response_loss = None, None
        if belief_labels is not None:
            assert response_labels is not None

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_belief_labels = belief_labels[..., 1:].contiguous()
            shift_response_labels = response_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            belief_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_belief_labels.view(-1))

            response_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_response_labels.view(-1))
            response_loss = response_ce

            bp_loss=0.
            ## we only use 0.5 weighted bp losses.
            if back_predict_labels is not None:
                shift_bp_labels = back_predict_labels[..., 1:].contiguous()
                bp_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_bp_labels.view(-1))
                # assert bp_weight is not None
                bp_loss*=bp_weight[0]
        
        output = (lm_logits, consistency_logits,) + transformer_outputs[1:]
        if consistency_loss is not None:
            output = (consistency_loss,) + output
        return ((belief_loss, response_loss + bp_loss, response_ce + bp_loss) + output) if belief_loss is not None else output


@dataclass
class ModelPredictor:
    model: transformers.PreTrainedModel = None
    tokenizer: transformers.PreTrainedTokenizer = None
    max_belief_length: int = 100
    max_response_length: int = 200
    device: torch.device = torch.device('cpu')

    @staticmethod
    def from_pretrained(model_name):
        config = transformers.GPT2Config.from_pretrained(model_name)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            model_name)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name, config=config)
        if model_name == 'gpt2':
            tokenizer, model = add_custom_tokens(tokenizer, model)
        tokenizer.pad_token = tokenizer.eos_token
        predictor = ModelPredictor(model, tokenizer)
        return predictor

    # def predict_belief(self, contexts):
    #     insert_labels = data.utils.InsertLabelsTransformation()
    #     tokenize = data.utils.TokenizerTransformation(
    #         self.tokenizer,
    #         max_context_length=self.model.config.n_ctx - self.max_belief_length - 1)
    #     eos_token_id = self.tokenizer.convert_tokens_to_ids(['<|eob|>'])[0]
    #     beliefs = []
    #     # TODO: batch generation
    #     for ctx in contexts:
    #         sample = insert_labels((ctx, None, None, None, 1))
    #         sample = tokenize.get_tokens(sample)[0]
    #         sample = torch.tensor(sample, dtype=torch.int64).to(self.device)
    #         sample = sample.view(1, *sample.shape)  # (batch, time)
    #         greedy_output = self.model.generate(
    #             input_ids=sample,
    #             max_length=sample.size(1) + self.max_belief_length,
    #             eos_token_id=eos_token_id,
    #             pad_token_id=eos_token_id,
    #             do_sample=False)
    #         # https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py

    #         prediction = greedy_output[0]
    #         offset = len(sample[0])
    #         prediction = prediction[:offset + (prediction[offset:] != eos_token_id).int().sum()]
    #         prediction = self.tokenizer.decode(prediction, skip_special_tokens=False,
    #                                            clean_up_tokenization_spaces=True)
    #         prefix = self.tokenizer.decode(sample[0], clean_up_tokenization_spaces=True) +\
    #             '=> ' + insert_labels.belief_label
    #         prediction = prediction[len(prefix):]
    #         beliefs.append(prediction)
    #     return beliefs

    def predict_response(self, dialogue_actions):
        insert_labels = data.utils.InsertLabelsTransformation()
        tokenize = data.utils.ScgptTransformation(
            self.tokenizer,
            max_context_length=self.model.config.n_ctx - self.max_response_length)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(['<|endoftext|>'])[0]
        responses = []
        # TODO: batch generation
        for acts in dialogue_actions:
            sample = insert_labels((None,None,None,None,None,None,None,None,None,None,acts,None))
            sample = tokenize.get_tokens(sample)[0]
            # print(self.tokenizer.decode(sample))
            sample = torch.tensor(sample, dtype=torch.int64).to(self.device)
            sample = sample.view(1, *sample.shape)  # (batch, time)
            greedy_output = self.model.generate(
                input_ids=sample,
                max_length=sample.size(1) + self.max_response_length,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
                do_sample=True,
                top_k=0)
            # https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py
            prediction = greedy_output[0]
            offset = len(sample[0])
            prediction = prediction[:offset + (prediction[offset:] != eos_token_id).int().sum()]
            prediction = self.tokenizer.decode(prediction, skip_special_tokens=False,
                                               clean_up_tokenization_spaces=True)
            prediction = prediction[len(self.tokenizer.decode(sample[0], clean_up_tokenization_spaces=True)):]
            prediction = prediction.lstrip()
            # print(prediction)
            responses.append(prediction)
        return responses

    def to(self, device):
        return dataclasses.replace(self, device=device, model=self.model.to(device))

class SCGPTModel(transformers.GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias",
                               r"lm\_head\.weight", r"binary\_head\.\w+"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = transformers.GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.consistency_head = nn.Linear(config.n_embd, 1) # ?
        self.auxiliary_dropout = nn.Dropout(config.summary_first_dropout)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self,
                input_ids=None,          # all input sequence tokens;
                past=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                consistency_token_ids=None, # the last token (eos),
                              # for classify whether consistent or not. 
                consistency_labels=None,  # is consistency or not
                user_intent_token_ids=None,
                user_intent_labels=None,
                user_intent_mask=None,
                belief_labels=None,      # context + belief states, and aims to predict bs.
                system_action_token_ids=None,
                system_action_labels=None,
                system_action_mask=None,
                response_labels=None, # only responses part has label, and others part is -100.
                back_predict_labels=None,  # this is the target of back predicted resonses.
                bp_weight=0.3,
                binary_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                **kwargs  # context=context_labels
                ):
        # print(f"shape of bp weight: {bp_weight.shape}")

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        def gather_auxiliary_features(token_ids):
            if token_ids is None:
                # torch.full_like(input,fill_value) returns the same size
                #as input with the filling of fill_value.
                
                # hidden_states.shape[-2] is max-seqence-length

                # ... in split means select the last dimension, so it means select the last
                # embedding dimension, and select the first batch, and with all seqence.
                # so the shape is 1*msl*1
                token_ids = torch.full_like(hidden_states[..., :1, :], # which the shape is ???
                                            hidden_states.shape[-2]-1, dtype=torch.long,)
            else:
                token_ids = token_ids.unsqueeze(-1).unsqueeze(-1)
                token_ids = token_ids.expand(
                    (-1,) * (token_ids.dim() - 1) + (hidden_states.size(-1),))

            # shape of binary_token_ids: (bsz, XX, 1, hidden_size)
            # where XX are optional leading dim of hidden_states
            # shape of binary_logits (bsz, XX, hidden_size)
            logits = hidden_states.gather(-2, token_ids).squeeze(-2)
            logits = self.auxiliary_dropout(logits)
            return logits

        # consistency_logits = self.consistency_head(gather_auxiliary_features(consistency_token_ids)).squeeze(-1)
        # consistency_loss = None
        # if consistency_labels is not None:
        #     # Auxiliary tasks
        #     aux_criterion = LabelSmoothingBCEWithLogitsLoss(self.config.summary_label_smoothing)
        #     consistency_loss = aux_criterion(consistency_logits, consistency_labels)

        belief_loss, response_loss = None, None
        if True:
            assert response_labels is not None

            shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_belief_labels = belief_labels[..., 1:].contiguous()
            shift_response_labels = response_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            # belief_loss = loss_fct(
            #     shift_logits.view(-1, shift_logits.size(-1)),
            #     shift_belief_labels.view(-1))

            # print("shift logits:{}".format(shift_logits.view(-1, shift_logits.size(-1)).shape))
            # print("shift response labels:{}".format(shift_response_labels.view(-1).shape))
            # print(shift_response_labels)

            response_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_response_labels.view(-1))
            # response_ce = loss_fct(shift_logits, shift_response_labels)
            response_loss = response_ce

            # bp_loss=0.
            # ## we only use 0.5 weighted bp losses.
            # if back_predict_labels is not None:
            #     shift_bp_labels = back_predict_labels[..., 1:].contiguous()
            #     bp_loss = loss_fct(
            #     shift_logits.view(-1, shift_logits.size(-1)),
            #     shift_bp_labels.view(-1))
            #     # assert bp_weight is not None
            #     bp_loss*=bp_weight[0]
        
        # output = (lm_logits, consistency_logits,) + transformer_outputs[1:]
        # if consistency_loss is not None:
        #     output = (consistency_loss,) + output
        # return ((belief_loss, response_loss + bp_loss, response_ce + bp_loss) + output) if belief_loss is not None else output
        return response_loss
