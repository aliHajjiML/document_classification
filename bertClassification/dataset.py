import torch

class BertClassificationDataset:
    def __init__(self, text, label, tokenizer, max_length):
        self.text =  text
        self.label = label
        self.tokenizer = tokenizer
        self.max_length = max_length
    #Length of dataset
    def __len__(self):
        return len(self.text)
    #
    def __getitem__(self, idx):

        #function applied for each example(item) of the dataset 
        text= str(self.text[idx])
        inputs = self.tokenizer.encode_plus(
                        text,                    # Sentence to encode.
                        add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                        return_attention_mask = True,   # Construct attn. masks.
                        max_length = self.max_length, # Sentence maximum length.
                        pad_to_max_length = True, # Pad sentence to maximum length
                        return_token_type_ids = False
                   )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids' : torch.tensor(inputs['input_ids'], dtype = torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype = torch.long),
            'label': torch.tensor(self.label[idx], dtype = torch.long)
        }

