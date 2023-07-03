import dataclasses
import logging
from dataclasses import dataclass
from transformers import GPT2Tokenizer as SoloistTokenizer  # noqa
from torch import nn
import transformers
from torch.nn import functional as F
import torch
import data
from typing import Optional,Tuple


EOB_TK = '<|eob|>'
EOKB_TK = '<|eokb|>'
EOT_TK = '<|endoftext|>'
SPECIAL_TOKENS = [EOB_TK, EOKB_TK]
logger = logging.getLogger()



## added by lz, it is from huggingface transformers
class AttentionModule(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class ControlLayer(nn.Module):
    def __init__(self, mrl:int,
                 temembed_dim:int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,):
        super().__init__()

        self.crossAttention=AttentionModule(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            is_decoder=True,
                                            bias=True)
        self.past_key_value=None

        # transform matrix for the embedding of templates.
        self.templateTransform=nn.Linear(temembed_dim,embed_dim)
        self.Softmax=nn.Softmax

    def forward(self,response_token_current,template_candidates):
        """
        shape of template-candidates: num_tem * max_tem_len * d_tem_embeddng
        shape of response, for training: max_r_len*d_embedding, or d_embedding in test.
        """
        ## aggregate template candidates.
        template_candidates=self.templateTransform(template_candidates)
        alpha=nn.Softmax(torch.matmul(template_candidates,response_token_current.T))
        # alpha shape: num_tem*max_tem_len*max_r_len
        # template representation shape: max_tem_len*max_r_len
        template_representation=torch.sum(torch.mul(template_candidates,alpha),dim=0)

        ## token-level cross attention
        if self.past_key_value is None:
            sattn_output,
            attn_weights_reshaped,
            past_key_value=self.crossAttention.forward(hidden_states=response_token_current,
                                        key_value_states=template_representation)
            # self.past_key_value=past_key_value
        else:
            sattn_output,
            attn_weights_reshaped,
            past_key_value=self.crossAttention.forward(hidden_states=response_token_current,
                                                       past_key_value=self.past_key_value)
        result=sattn_output+response_token_current

        return result
        

class InfoLayer(nn.Module):
    def __init__(self, mrl:int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,):
        super().__init__()

        self.crossAttention=AttentionModule(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            is_decoder=True,
                                            bias=True)
        self.past_key_value=None

    def forward(self,response_token_current,dialog_states):
        if self.past_key_value is None:
            sattn_output,
            attn_weights_reshaped,
            past_key_value=self.crossAttention.forward(hidden_states=response_token_current,
                                                       key_value_states=dialog_states)
            self.past_key_value=past_key_value
        else:
            sattn_output,
            attn_weights_reshaped,
            past_key_value=self.crossAttention.forward(hidden_states=response_token_current,
                                                       past_key_value=self.past_key_value)
        result=sattn_output+response_token_current
        return result
            

class MyModel(transformers.GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias",
                               r"lm\_head\.weight", r"binary\_head\.\w+"]

    def __init__(self, config):
        super().__init__(config)
        
        self.controlLayer=ControlLayer(mrl=config.n_ctx,
                 temembed_dim=300,
                 embed_dim=768,
                 num_heads=12,dropout=0.1)

        self.InfoLayer=InfoLayer(mrl=config.n_ctx,
                 embed_dim=config.n_embd,
                 num_heads=config.n_head,
                                 dropout=config.attn_pdrop)

        self.tokenEmbed2temEmbed=nn.Linear(config.n_embd,200)
        self.temEmbed2tokenEmbed=nn.Linear(200,config.n_embd)

        self._load_retrieval()

        self.transformer = transformers.GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.consistency_head = nn.Linear(config.n_embd, 1) 
        self.auxiliary_dropout = nn.Dropout(config.summary_first_dropout)
        self.init_weights()

    def _load_templates(self):
        with open(self.args.inducted_template_list_path,'rb') as f:
            self.inducted_templates=pickle.load(f)

    def _load_retrieval(self):
        from retrievalWord2Vec import retrievalModel as retrieval
        self.retrieval=retrieval()
        self.logger.info("RETRIEVAL MODEL LOAD DONE.")

    def get_output_embeddings(self):
        return self.lm_head

    def findResponseBeginIndex(self,indexes):
        for index in indexes:
            if index==50258:
                return index
        return indexes[-1]

    def findDialogueStateBeginIndex(self,indexes):
        for index in indexes:
            if index==50256:
                return index
        return indexes[0]

    def forward(self,
                input_ids=None,
                past=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                consistency_token_ids=None,
                consistency_labels=None,
                user_intent_token_ids=None,
                user_intent_labels=None,
                user_intent_mask=None,
                belief_labels=None,
                system_action_token_ids=None,
                system_action_labels=None,
                system_action_mask=None,
                response_labels=None,
                binary_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,

                # candidate_templates=None,
                using_template_model=True,
                template_label=None,
                using_info_model=True,
                use_state_back_prediction=False,
                negative_states=None,
                use_template_back_prediction=False,
                **kwargs
                ):

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
        r_id=self.findResponseBeginIndex(input_ids)
        s_id=self.findDialogueStateBeginIndex(input_ids)

        hidden_states = transformer_outputs[0]
        
        response_states=hidden_states[r_id:]
        dialog_state_states=hidden_states[s_id:r_id]

        if using_template_model:
            previous_info=hidden_states[r_id-1]
            previous_info=tokenEmbed2temEmbed(previous_info)

            # retrieval loss 
            inductedEmbeddings=self.retrieval.getTemplateMatrix(self.retrieval.inducted_templates)
            distribution=torch.consine_similarity(previous_info,inductedEmbeddings)
            maxinum=torch.argmax(distribution)
            control_loss1=nn.CrossEntropyLoss(distribution,template_label)

            # search loss
            indexes=self.retrieval.searchIndex(previous_info,k=3)
            candidate_embeds=self.retrieval.Index2WordEmbeddingBatch(indexes)
            candidate_templates=torch.tensor(candidate_embeds)
            candidate_templates=self.temEmbed2tokenEmbed(candidate_templates)

            tem_result=self.ControlLayer(response_states,candidate_templates)

            hidden_states[r_id:]=hidden_states[r_id:]+tem_result
        if using_template_model:
            info_result=self.InfoLayer(response_states,dialog_state_states)
            hidden_states[r_id:]=hidden_states[r_id:]+info_result

        lm_logits = self.lm_head(hidden_states)
        # return lm_logits

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

        consistency_loss = None
        consistency_logits = self.consistency_head(gather_auxiliary_features(consistency_token_ids)).squeeze(-1)
        output = (lm_logits, consistency_logits,) + transformer_outputs[1:]
        # consistency loss
        if consistency_labels is not None:
            # Auxiliary tasks
            aux_criterion = LabelSmoothingBCEWithLogitsLoss(self.config.summary_label_smoothing)
            consistency_loss = aux_criterion(consistency_logits, consistency_labels)

        if consistency_loss is not None:
            output = (consistency_loss,) + output
            
        # belief states loss and response_loss
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

            output=(belief_loss, response_loss, response_ce) + output

        # back prediction loss
        state_bp_loss=0.
        response_representation=torch.max(response_states)
        if use_state_back_prediction:
            all_states=negative_states.expand(previous_info)
            state_bp_loss=-1*nn.Softmax(torch.matmul(all_states,response_representation.T))[-1]
        tem_bp_loss=0.
        if use_template_back_prediction:
            negative_templates=inductedEmbeddings[torch.randint(len(negative_states))]
            all_states=negative_templates.expand(inductedEmbeddings[template_label])
            state_bp_loss=-1*nn.Softmax(torch.matmul(all_states,response_representation.T))[-1]

        output = (state_bp_loss,tem_bp_loss,) + output
            
        return output

if __name__=="__main__":
    # load mymodel
    mymodel=MyModel()
