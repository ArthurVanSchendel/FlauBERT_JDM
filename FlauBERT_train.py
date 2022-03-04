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

def raw_txt_to_txt(filename, data_type, output_name):
  file = open(filename, 'rt')
  text = file.readlines()
  file.close()
  text = np.array(text)
  print("TEXT SHAPE = ", text.shape)

  txt_data = pd.DataFrame(columns=['aggregate'])
  aggregates = []
  for i in range(text.shape[0]):
    if text[i] == "\n":
      continue
    splitted_text = text[i].split(";")
    splitted_text = np.array(splitted_text)
    splitted_text[0] = splitted_text[0].strip()
    splitted_text[0] = splitted_text[0] + '.'
    aggregates.append(splitted_text[0])
  txt_data['aggregates'] = aggregates
  txt = txt_data.to_csv(f"{output_name}.txt", encoding="utf-8-sig")
  return txt


def extract_txt_from_html(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    final_txt = []
    soup.prettify()
    sentences = soup.find_all('sen')
    for i in range(len(sentences)):
      text = str(sentences[i]).replace("<sen>", '')
      text = text.replace("</sen>", '.')
      final_txt.append(text)
    return final_txt


def create_dataset(nb_calls, url, type_dataset):
  final_dataset = pd.DataFrame(columns=["sentences"])
  dataset = []
  for i in range(nb_calls):
    strip = extract_txt_from_html(url)
    dataset.append(strip)
  final_dataset['sentences'] = dataset
  dataset_txt = final_dataset.to_csv(f"aggregate_{type_dataset}.txt", encoding="utf-8-sig")
  return dataset_txt

modelname = 'flaubert/flaubert_large_cased'
tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(modelname)

url_data = "http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=100"
#### THIS LINK GENERATES 1300 LINES (SENTENCES)

txt_train = create_dataset(3, url_data, "train")
txt_valid = create_dataset(2, url_data, "valid")
#txt_train = raw_txt_to_txt('input_raw_tr.txt', 'aggregates', 'aggregate_train')
#txt_valid = raw_txt_to_txt('input_raw_valid.txt', 'aggregates', 'aggregate_valid')

## LOAD DATASETs HERE ###
datasets = load_dataset('text', data_files={'train': 'aggregate_train.txt', 'validation': 'aggregate_valid.txt'})
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=2, remove_columns=['text'])

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=512,
    num_proc=2,
)

print("DATASETS TRAIN = ", datasets['train'])
print("DATASETS VALID = ", datasets['validation'])

model_name = modelname.split("/")[-1]

training_args = TrainingArguments(
    f"{model_name}-finetuned-JDM_text",
    evaluation_strategy = "epoch",
    learning_rate=6e-5,
    num_train_epochs=5,
    logging_steps=50,
    per_device_train_batch_size=8,
    weight_decay=0.02,
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
