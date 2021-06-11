import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import re
import unidecode
import wordninja


class Data_Loader():

	def __init__(self, data_dir, file_path, model_name, split_ratio, batch_size = 32, process_type = "training"):


		self.process_type = process_type
		self.data_dir = data_dir
		self.file_path = file_path
		if model_name == "bert-base-uncased":
			self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = True)
		elif model_name == "roberta-base":
			self.tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case = True)
		self.split_ratio = split_ratio
		self.batch_size = batch_size

	def max_length(self,sentences):
		max_len = 0
		for sent in sentences:
			input_ids = self.tokenizer.encode(sent)

			max_len = max(max_len, len(input_ids))

		return max_len

	def clean_tweet(self, text):
    
	    # lower-case all characters
	    text=text.lower()
	    
	    # remove twitter handles
	    text= re.sub(r'@\S+', '',text) 
	    
	    # remove urls
	    text= re.sub(r'http\S+', '',text) 
	    text= re.sub(r'pic.\S+', '',text)
	      
	    # replace unidecode characters
	    text=unidecode.unidecode(text)
	      
	    # regex only keeps characters
	    text= re.sub(r"[^a-zA-Z+']", ' ',text)
	    
	    # keep words with length>1 only
	    text=re.sub(r'\s+[a-zA-Z]\s+', ' ', text+' ') 

	    # split words like 'whatisthis' to 'what is this'
	    def preprocess_wordninja(sentence):      
	        def split_words(x):
	            x=wordninja.split(x)
	            x= [word for word in x if len(word)>1]
	            return x
	        new_sentence=[ ' '.join(split_words(word)) for word in sentence.split() ]
	        return ' '.join(new_sentence)
	    
	    text=preprocess_wordninja(text)
	    
	    # regex removes repeated spaces, strip removes leading and trailing spaces
	    text= re.sub("\s[\s]+", " ",text).strip()  
	    
	    return text


	def map_word_ids(self, sentences, labels):
		input_ids = []
		attention_masks = []

		for sent in sentences:
			# `encode_plus` will:
		    #   (1) Tokenize the sentence.
		    #   (2) Prepend the `[CLS]` token to the start.
		    #   (3) Append the `[SEP]` token to the end.
		    #   (4) Map tokens to their IDs.
		    #   (5) Pad or truncate the sentence to `max_length`
		    #   (6) Create attention masks for [PAD] tokens.

		    encoded_dict = self.tokenizer.encode_plus(
	    											sent,
	    											add_special_tokens = True,
	    											max_length = 64,
	    											pad_to_max_length = True,
	    											return_attention_mask = True,
	    											return_tensors = 'pt')

		    input_ids.append(encoded_dict['input_ids'])
		    attention_masks.append(encoded_dict['attention_mask'])

		input_ids = torch.cat(input_ids, dim = 0)
		attention_masks = torch.cat(attention_masks, dim = 0)
		labels = torch.tensor(labels)

		return encoded_dict, input_ids, attention_masks, labels


	def train_validation_split(self, input_ids, attention_masks, labels, split_ratio):
		dataset = TensorDataset(input_ids, attention_masks, labels)
		train_size = int(split_ratio*len(dataset))
		val_size = len(dataset) - train_size

		train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

		print('{:>5,} training samples'.format(train_size))
		print('{:>5,} validation samples'.format(val_size))

		return train_dataset, val_dataset


	def read_data(self):

		df = pd.read_csv(self.data_dir + self.file_path)
		df['tweet'] = df['tweet'].apply(lambda x: self.clean_tweet(x))
		sentences = list(df['tweet'].values)
		labels = df.label.values
		max_len = self.max_length(sentences)

		# max_len in the training set is 65 and in test set it is 63.

		encoded_dict, input_ids, attention_masks, labels = self.map_word_ids(sentences, labels)

		if self.process_type == "training":
			train_dataset, val_dataset = self.train_validation_split(input_ids, attention_masks, labels, self.split_ratio)

			train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = self.batch_size)
			val_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = self.batch_size)

			full_train_data = TensorDataset(input_ids, attention_masks, labels)
			full_train_sampler = RandomSampler(full_train_data)
			full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size = self.batch_size)

			return train_dataloader, val_dataloader, full_train_dataloader

		elif self.process_type == "inferencing":
			prediction_data = TensorDataset(input_ids, attention_masks, labels)
			prediction_sampler = SequentialSampler(prediction_data)
			prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size = self.batch_size)

			return prediction_dataloader