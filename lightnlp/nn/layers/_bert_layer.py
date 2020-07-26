import torch
import torch.nn as nn
from transformers import BertModel


class BertLayer(nn.Module):
    """
    This layer is callable for bert from pre trained model to generate sent emb or emb list
    """
    def __init__(self, pre_trained_model_path, finetune=True):
        super(BertLayer, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_path)
        self.config = self.bert.config
        self.finetune = finetune

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                ):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        seq_out = outputs[0]
        last_out = outputs[1]
        return seq_out, last_out
