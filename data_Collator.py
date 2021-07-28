from dataclasses import dataclass
from typing import Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input： batch * [original_tokens: tensor, masked_tokens: tensor, mask_label: tensor]
        # original_tokens, masked_tokens, mask_label = zip(*examples)
        masked_tokens, mask_label = zip(*examples)
        inputs = self._tensorize_batch(masked_tokens) # masked_sequence_ids
        labels = self._tensorize_batch(mask_label) # mask_label with PAD_TOK

        # Clear PAD_TOK
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id) # all elements of PAD_ID is TRUE, else FALSE
            labels.masked_fill_(padding_mask, -100)

        # input_ids: Batch * [CLS_id, id_1, ..., ]
        # labels: Batch * [-100, ..., MASKED_TOK_ID, ...., -100]
        return {"input_ids": inputs, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        """
        这个东西现在可以直接用了
        """
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
