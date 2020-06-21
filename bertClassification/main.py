import pandas as pd
import dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, AdamW
from model import BertTextClassification
import training
import torch

def main():

	# Read data
	df = pd.read_csv("../data/thedeep.data.txt", sep=",", header=1, names=['sentence_id', 'text', 'label'])

	# Load the BERT tokenizer
	print('Loading BERT Tokenizer....')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


	# Use model selection library to split data: training data and validating data
	training_inputs, validation_inputs = train_test_split(df, random_state= 2018, test_size= 0.3)
	batch_size = 32
	max_length = 80
	Epochs = 1


	train_dataset = dataset.BertClassificationDataset(
    	text = training_inputs.text.values,
    	label = training_inputs.label.values,
    	tokenizer = tokenizer,
    	max_length = max_length
    	)
	valid_dataset = dataset.BertClassificationDataset(
    	text = validation_inputs.text.values,
    	label = validation_inputs.label.values,
    	tokenizer = tokenizer,
    	max_length = max_length
    	)

	# Create a dataloader for training and validation data
	train_dataloader = DataLoader(train_dataset, sampler= RandomSampler(train_dataset), batch_size= batch_size)
	valid_dataloader = DataLoader(valid_dataset, sampler= RandomSampler(valid_dataset), batch_size= batch_size)


	# Create a instance of bert model, optimizer and scheduler
	device = torch.device("cuda")
	print('Loading BERT Model....')
	model = BertTextClassification('bert-base-uncased')
	model = model.to(device)

	param_optimizer = list(model.named_parameters())


	no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias'] 
	optimizer_grouped_parameters = [
	    	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
	     	'weight_decay_rate': 0.1},
	    	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
	     	'weight_decay_rate': 0.0}
		]

	num_train_steps = int(len(train_dataset)*Epochs)
	optimizer = AdamW(optimizer_grouped_parameters, lr= 3e-5)
	scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps = 0,
			num_training_steps = num_train_steps
			)
	loss_fn = nn.CrossEntropyLoss().to(device)


	# Training and evaluating the model on the validation dataset
	training_stats = []
	best_accuracy = 0

	for epoch in range(Epochs):

	  print(f'Epoch {epoch + 1}/{Epochs}')
	  print('-' * 10)

	  train_accuracy, train_loss = training.train_epoch(model, train_dataloader, loss_fn, optimizer, device, scheduler, training_inputs)

	  print(f'Train loss {train_loss} accuracy {train_accuracy}')

	  val_accuracy, val_loss = training.eval_model(model, valid_dataloader, loss_fn, device, validation_inputs)

	  print(f'Val   loss {val_loss} accuracy {val_accuracy}')
	  print()

	  # Record all statistics from this epoch.
	  training_stats.append(
	        {
	            'epoch': epoch + 1,
	            'Training Loss': train_loss,
	            'Training Accuracy': train_accuracy,
	            'Valid. Loss': val_loss,
	            'Valid. Accur.': val_accuracy,
	        }
	    )

	  if val_accuracy > best_accuracy:
	    torch.save(model.state_dict(), 'best_model.bin')
	    best_accuracy = val_accuracy



if __name__ == "__main__":

    main() 