import torch.nn as nn
import transformers


class BertTextClassification(nn.Module):

    def __init__(self, pretrained_model):
        super(BertTextClassification, self).__init__()
        self.pretrained_model = pretrained_model
        self.bert = transformers.BertModel.from_pretrained(self.pretrained_model)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(768, 12)

    def forward(self, input_ids, attention_mask):

        last_hidden_state, pooler_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask)

        pooler_output_dropout =  self.dropout(pooler_output)
        output = self.output(pooler_output_dropout)

        return output