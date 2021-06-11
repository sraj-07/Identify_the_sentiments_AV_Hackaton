from dataloader import Data_Loader
import torch
import argparse
import pandas as pd
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
import numpy as np
import torch.nn.functional as F




def load_checkpoint(load_path, model, device):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def process_csv(test_file, target = "label"):
	df = pd.read_csv(test_file)
	if target not in df.columns:
		df[target] = 0 
		df = df[['id','label','tweet']] 
		df.to_csv('/notebooks/sentiment_analysis/'+test_file, index = False)
	else:
		pass




def evaluate(model, test_loader, device):
	# y_true = []
	y_pred = []
	probabs = []

	total_eval_loss = 0
	model.eval()
	for batch in test_loader:
		b_input_ids = batch[0].to(device)
		b_input_mask = batch[1].to(device)
		b_labels = batch[2].to(device)



		with torch.no_grad():


			(loss, logits) = model(b_input_ids,
									token_type_ids = None,
									attention_mask = b_input_mask,
									labels = b_labels)


		total_eval_loss += loss.item()


		probabs.extend(F.softmax(logits).cpu().numpy())

		logits = logits.detach().cpu().numpy()

		pred_flat = np.argmax(logits, axis=1).flatten()

		y_pred.extend(pred_flat)


	# total_eval_accuracy += flat_accuracy(logits, label_ids)

	return y_pred, probabs


def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--data_dir", default="", type=str)

	parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)

	parser.add_argument("--test_file", default="test.csv", type=str)
	
	parser.add_argument("--split_ratio", default=0.8, type=float)

	parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training.")


	
	args = parser.parse_args()
	print(args)

	if args.model_name_or_path == "bert-base-uncased":
		model = BertForSequenceClassification.from_pretrained(
				    args.model_name_or_path, # Use the 12-layer BERT model, with an uncased vocab.
				    num_labels = 2, # The number of output labels--2 for binary classification.
				                    # You can increase this for multi-class tasks.   
				    output_attentions = False, # Whether the model returns attentions weights.
				    output_hidden_states = False, # Whether the model returns all hidden-states.
				    )
	elif args.model_name_or_path == "roberta-base":
		model = RobertaForSequenceClassification.from_pretrained(
				    args.model_name_or_path, # Use the 12-layer BERT model, with an uncased vocab.
				    num_labels = 2, # The number of output labels--2 for binary classification.
				                    # You can increase this for multi-class tasks.   
				    output_attentions = False, # Whether the model returns attentions weights.
				    output_hidden_states = False, # Whether the model returns all hidden-states.
				    )

	data_loader = Data_Loader(args.data_dir, args.test_file, args.model_name_or_path, args.split_ratio, process_type = "inferencing")

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	process_csv(args.test_file)
	load_checkpoint('/notebooks/sentiment_analysis/' + '/model_full_train.pt', model, device)
	model.to(device)
	test_loader = data_loader.read_data()
	out = evaluate(model, test_loader, device)
	df = pd.read_csv(args.test_file)
	df['label'] = out[0]
	df['probabs'] = out[1]
	df[['id','probabs']].to_csv("inference_result_probabs.csv", index = False)
	df[['id','label']].to_csv("inference_result.csv", index = False)


if __name__ == '__main__':
	main()