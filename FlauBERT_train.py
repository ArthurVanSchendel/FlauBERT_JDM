from posixpath import join
import torch
import transformers
import numpy as np
import pandas as pd
import csv
import math
from urllib.request import urlopen
from bs4 import BeautifulSoup


from transformers import XLMTokenizer, XLMWithLMHeadModel, pipeline
from transformers import FlaubertModel, FlaubertTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from datasets import load_dataset


if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.cuda.current_device()
else:
    print("Will work on CPU.")

def tokenize_function(examples):
    return tokenizer(examples["text"])

block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def extract_txt_from_html(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    soup.prettify()
    indexes = []
    sentences = soup.find_all('sen')
    lex_sem_relations = soup.find_all('ch')
    print("soup = ", soup)
    for j in range(len(sentences)):
      indexes.append(j)
    final_txt = pd.DataFrame(columns=['sentences', 'lexico_sem_relation'], index=indexes)
    for i in range(len(sentences)):
      tmp_txt = pd.DataFrame(columns=['sentences', 'lexico_sem_relation'], index=indexes)
      text = str(sentences[i]).replace('<sen>', '')
      text = text.replace('</sen>', '.')
      lex_sem = str(lex_sem_relations[i]).replace('<ch>', '')
      lex_sem = lex_sem.replace('</ch>', '')
      lex_sem = lex_sem.replace('&gt;', '>')

      final_txt['sentences'].loc[i] = text
      final_txt['lexico_sem_relation'].loc[i] = lex_sem
      print("TEXT SHAPE = ", np.array(text).shape)
      print("Extract txt from html = ", final_txt)
      print("LEN SENTENCES = ", len(sentences))
    return final_txt, len(sentences)


def create_dataset(nb_calls, url, type_dataset):
  indexes = []
  final_dataset = pd.DataFrame(columns=['sentences', 'lexico_sem_relation'])
  #tmp_dataset = []
  for i in range(nb_calls):
    strip, len_data = extract_txt_from_html(url)
    print("STRIP BEFORE APPEND TO DATASET = ", strip)
    final_dataset = pd.concat([final_dataset, strip])

  print('FINAL DATASET = ', final_dataset)
  final_dataset.to_csv(f"aggregate_{type_dataset}.txt", encoding="utf-8-sig", columns=['sentences'])
  return final_dataset

modelname = 'flaubert/flaubert_large_cased'
tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(modelname)


######   DATASET PARAMETERS #######

##  CHUNK = Nombre de chunk selectionné
## ITER  = Nombre de variantes par chunk

### TAILLE MAX DE DATASET = CHUNK * ITER


url_data = "http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=10&verbose=0&iter=100"
#### THIS LINK GENERATES 1300 LINES (SENTENCES)

txt_train = create_dataset(2, url_data, "train")
txt_valid = create_dataset(1, url_data, "valid")
#txt_train = raw_txt_to_txt('input_raw_tr.txt', 'aggregates', 'aggregate_train')
#txt_valid = raw_txt_to_txt('input_raw_valid.txt', 'aggregates', 'aggregate_valid')
print("txt train = ", txt_train)
print("txt train = ", txt_valid)
## LOAD DATASETs HERE ###
datasets = load_dataset('text', data_files={'train': 'aggregate_train.txt', 'validation': 'aggregate_valid.txt'})
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=['text'])

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

print("DATASETS TRAIN = ", datasets['train'])
print("DATASETS VALID = ", datasets['validation'])
print("len dataset = ", np.array(datasets).shape)

model_name = modelname.split("/")[-1]

training_args = TrainingArguments(
    f"{model_name}-finetuned-JDM_text",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    num_train_epochs=7,
    logging_steps=1,
    per_device_train_batch_size=8,
    weight_decay=0.01,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)  ###  15% of sentence is masked

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)
####### TRAIN ###########
trainer.train()
####### EVAL ############
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
