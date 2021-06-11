# Identify_the_sentiments_AV_Hackaton

My solution to solve a simple practice NLP classification challenge on Analytics Vidhya's website - Identify the Sentiments.

Problem statement -
Sentiment analysis remains one of the key problems that has seen extensive application of natural language processing. This time around, given the tweets from customers about various tech firms who manufacture and sell mobiles, computers, laptops, etc, the task is to identify if the tweets have a negative sentiment towards such companies or products.

Dataset
The train set contains 7,920 tweets The test set contains 1,953 tweets

Data Pre-processing
(Standard Steps taken from here - Link)

Lower-case all characters

Remove twitter handles

Remove urls

Replace unidecode characters

Only keep characters

Keep words with length>1 only

Replace words like 'whatisthis' to ' what is this'

Remove repeated spaces

Approach

Take a bert-base-uncased pre-trained transformer model to create embeddings and finetune it for the Sequence Classification Task.
Got a public LB of 0.91xx using this approach


Take a roberta-base pre-trained transformer model to create embeddings and finetune it for the Sequence Classification Task.
Got a public LB of 0.909x using this approach
