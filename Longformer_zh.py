from transformers import BertForMaskedLM, BertTokenizerFast, BertConfig
from transformers.modeling_longformer import LongformerSelfAttention

""" 
RobertaLong
    RobertaLongForMaskedLM: the long version of the ROBERTa model.
    Replaces BertSelfAttention with RoberaLongSelfAttention
"""
class BertLongSelfAttention(LongformerSelfAttention):
    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False
                ):
        return super().forward(hidden_states,
                               attention_mask=attention_mask,
                               output_attentions=output_attentions)

# change it into bert-version
class LongformerZhForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        # Replace self-attention with long-attention
        for i, layer in enumerate(self.bert.encoder.layer):
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)