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
from transformers import CamembertModel, CamembertTokenizer
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
    for j in range(len(sentences)):
      indexes.append(j)
    final_txt = pd.DataFrame(columns=['sentences', 'lexico_sem_relation'], index=indexes)
    #final_txt = pd.DataFrame(columns=['sentences'])
    for i in range(len(sentences)):
      tmp_txt = pd.DataFrame(columns=['sentences', 'lexico_sem_relation'], index=indexes)
      text = str(sentences[i]).replace('<sen>', '')
      text = text.replace('</sen>', '.')
      lex_sem = str(lex_sem_relations[i]).replace('<ch>', '')
      lex_sem = lex_sem.replace('</ch>', '')
      lex_sem = lex_sem.replace('&gt;', '>')

      final_txt['sentences'].loc[i] = text
      final_txt['lexico_sem_relation'].loc[i] = lex_sem
    return final_txt, len(sentences)


def create_dataset(nb_calls, url, type_dataset):
  indexes = []
  final_dataset = pd.DataFrame(columns=['sentences', 'lexico_sem_relation'])
  #tmp_dataset = []
  for i in range(nb_calls):
    strip, len_data = extract_txt_from_html(url)
    final_dataset = pd.concat([final_dataset, strip])
  final_dataset.to_csv(f"aggregate_{type_dataset}.txt", encoding="utf-8-sig", columns=['sentences'])
  return final_dataset

modelname1 = 'flaubert/flaubert_large_cased'
modelname2 = 'flaubert/flaubert_base_cased'
modelname3 = 'flaubert/flaubert_base_uncased'
modelname4 = 'flaubert/flaubert_small_cased'

modelname5 = 'camembert-base'

#modelname6 = 'camembert/camembert-large'
#modelname7 = 'camembert/camembert-base-ccnet'
#modelname8 = 'camembert/camembert-base-wikipedia-4gb'

modelname9 = 'bert-base-multilingual-cased'

models = [modelname1, modelname2, modelname3, modelname4, modelname5, modelname9]  ## remove model_name_6 = have to convert from slow to fast token

tokens = []
mods = []
for name_model in models:
  print("MODEL NAME = ", name_model)

  if (name_model == 'camembert/camembert-large') or (name_model == 'camembert/camembert-base-ccnet') or (name_model=='camembert/camembert-base-wikipedia-4gb'):
    tokenizer = CamembertTokenizer.from_pretrained(name_model)
    model = CamembertModel.from_pretrained(name_model)
  else:
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    model = AutoModelForMaskedLM.from_pretrained(name_model)
  tokens.append(tokenizer)
  mods.append(model)

#tokenizer1 = AutoTokenizer.from_pretrained(modelname1, use_fast=True)
#model = AutoModelForMaskedLM.from_pretrained(modelname)


######   DATASET PARAMETERS #######

##  CHUNK = Nombre de chunk selectionné
## ITER  = Nombre de variantes par chunk

### TAILLE MAX DE DATASET = CHUNK * ITER


url_data = "http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=2"
#### THIS LINK GENERATES 1300 LINES (SENTENCES)

txt_train = create_dataset(2, url_data, "train")
txt_valid = create_dataset(1, url_data, "valid")
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

model_name1 = modelname1.split("/")[-1]
model_name2 = modelname2.split("/")[-1]
model_name3 = modelname3.split("/")[-1]
model_name4 = modelname4.split("/")[-1]
model_name5 = modelname5.split("/")[-1]
model_name9 = modelname9.split("/")[-1]

perplexity_results = []
min_perplex = 999.9
min_index = -1

i=0
for model in models:

  training_args = TrainingArguments(
      f"{model}-finetuned-JDM_text",
      evaluation_strategy = "epoch",
      learning_rate=2e-5,
      num_train_epochs=7,
      logging_steps=1,
      per_device_train_batch_size=8,
      weight_decay=0.01,
  )
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokens[i], mlm_probability=0.15)  ###  15% of sentence is masked

  trainer = Trainer(
      model=mods[i],
      args=training_args,
      train_dataset=lm_datasets["train"],
      eval_dataset=lm_datasets["validation"],
      data_collator=data_collator,
  )
  ####### TRAIN ###########
  print(f"BEGINNING TRAIN FOR {model}")
  trainer.train()
  ####### EVAL ############
  print(f"BEGINNING EVAL FOR {model}")
  eval_results = trainer.evaluate()
  perplexity = math.exp(eval_results['eval_loss'])
  if perplexity < min_perplex:
    min_perplex = perplexity
    min_index = i
  print(f"Perplexity: {perplexity}")
  perplexity_results.append(perplexity)
  i+=1

print(f"FINISHED TRAINING OF MODELS : {models}")
print(f"results : {perplexity_results}")
print(f"MIN PERPLEXITY : {min_perplex} for model {models[min_index]}")
