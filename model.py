from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer, RobertaTokenizer
import argparse
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import time
import datetime
from dataloader import Data_Loader
import torch



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def train(model, train_dataloader, optimizer, scheduler, device, valid_dataloader=None,epochs=2, evaluation=True):

	training_stats = []

	total_t0 = 0

	best_valid_loss = float("Inf")

	for epoch_i in range(0, epochs):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Training...')

		t0 = time.time()

		total_train_loss = 0

		model.train()

		for step, batch in enumerate(train_dataloader):

			if step%40 == 0 and not step == 0:

				elapsed = format_time(time.time() - t0)

				print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)

			model.zero_grad()

			loss, logits = model(b_input_ids,
    							token_type_ids = None,
    							attention_mask = b_input_mask,
    							labels = b_labels)

			total_train_loss += loss.item()

			# Perform a backward pass to calculate the gradients.

			loss.backward()

    		# Clip the norm of the gradients to 1.0.
	        # This is to help prevent the "exploding gradients" problem.

			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			optimizer.step()

			scheduler.step()


		avg_train_loss = total_train_loss/len(train_dataloader)

		training_time = format_time(time.time() - t0)

		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epcoh took: {:}".format(training_time))


		if evaluation == True:
			# ========================================
			#               Validation
			# ========================================
			# After the completion of each training epoch, measure our performance on
			# our validation set.


			print("")
			print("Running Validation...")

			t0 = time.time()

			model.eval()

			total_eval_accuracy = 0
			total_eval_loss = 0
			nb_eval_steps = 0

			for batch in valid_dataloader:

				b_input_ids = batch[0].to(device)
				b_input_mask = batch[1].to(device)
				b_labels = batch[2].to(device)



				with torch.no_grad():


					(loss, logits) = model(b_input_ids,
											token_type_ids = None,
											attention_mask = b_input_mask,
											labels = b_labels)


				total_eval_loss += loss.item()


				logits = logits.detach().cpu().numpy()

				label_ids = b_labels.to('cpu').numpy()


				total_eval_accuracy += flat_accuracy(logits, label_ids)

			# Report the final accuracy for this validation run.
			avg_val_accuracy = total_eval_accuracy / len(valid_dataloader)
			print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

			# Calculate the average loss over all of the batches.
			avg_val_loss = total_eval_loss / len(valid_dataloader)

			if best_valid_loss > avg_val_loss:
				# print("----saving model------at-----",file_path)
				best_valid_loss = avg_val_loss
				save_checkpoint("/notebooks/sentiment_analysis" + '/' + 'model.pt', model, best_valid_loss)

			# Measure how long the validation run took.
			validation_time = format_time(time.time() - t0)

			print("  Validation Loss: {0:.2f}".format(avg_val_loss))
			print("  Validation took: {:}".format(validation_time))

			# Record all statistics from this epoch.
			training_stats.append(
				{
				'epoch': epoch_i + 1,
				'Training Loss': avg_train_loss,
				'Valid. Loss': avg_val_loss,
				'Valid. Accur.': avg_val_accuracy,
				'Training Time': training_time,
				'Validation Time': validation_time
				}
			)

		else:
			save_checkpoint("/notebooks/sentiment_analysis" + '/' + 'model_full_train.pt', model, avg_train_loss)


		print("")
		print("Training complete!")

		print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--data_dir", default="", type=str)

	parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)

	parser.add_argument("--train_file", default="train.csv", type=str)
	
	parser.add_argument("--split_ratio", default=0.8, type=float)

	parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size for training.")
	parser.add_argument("--lr", default=3e-5, type=float,
                        help="learning rate for training.")
	parser.add_argument("--num_labels", default=1, type=int,
                        help="Max number of labels in the prediction.")
	parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
	parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization.")
	parser.add_argument("--process_type", type=str, default="Training",
                        help="random seed for initialization.")
	args = parser.parse_args()
	print(args)

	seed_val = args.seed

	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	data_loader = Data_Loader(args.data_dir, args.train_file, args.model_name_or_path, args.split_ratio, process_type = "training", batch_size = args.batch_size)

	train_dataloader, valid_dataloader, full_train_dataloader = data_loader.read_data()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
				    args.model_name_or_path, # Use the 12-layer roberta model, with an uncased vocab.
				    num_labels = 2, # The number of output labels--2 for binary classification.
				                    # You can increase this for multi-class tasks.   
				    output_attentions = False, # Whether the model returns attentions weights.
				    output_hidden_states = False, # Whether the model returns all hidden-states.
				    )

	model = model.to(device)

	optimizer = AdamW(model.parameters(),
                  lr = args.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
	epochs = args.num_train_epochs

	total_steps = len(train_dataloader)*epochs

	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

	

	train(model, train_dataloader, optimizer, scheduler, device, valid_dataloader=valid_dataloader,epochs=args.num_train_epochs, evaluation=True)


	print("Training model on full training data")

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

	model = model.to(device)

	optimizer = AdamW(model.parameters(),
                  lr = args.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
	epochs = args.num_train_epochs

	total_steps = len(full_train_dataloader)*epochs

	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

	train(model, full_train_dataloader, optimizer, scheduler, device,epochs=args.num_train_epochs, evaluation=False)




if __name__ == '__main__':
	main()