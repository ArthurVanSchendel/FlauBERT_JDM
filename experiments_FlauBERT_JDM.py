import torch
import transformers
import numpy as np
import pandas as pd
import csv
import math

from transformers import XLMTokenizer, XLMWithLMHeadModel, pipeline
from transformers import FlaubertModel, FlaubertTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from datasets import load_dataset


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
  return txt_data

def raw_txt_to_csv(filename, data_type, output_name):
  file = open(filename, 'rt')
  text = file.readlines()
  file.close()
  text = np.array(text)
  print("TEXT SHAPE = ", text.shape)

  #csv_data = pd.DataFrame(columns=[f'{data_type}', 'lexico_semantic_relation'])
  csv_data = pd.DataFrame(columns=['aggregates', 'lexico_semantic_relation'])
  aggregates = []
  lexico_semantic_relations = []

  for i in range(text.shape[0]):
    if text[i] == '\n':
      continue
    splitted_txt = text[i].split(";")
    splitted_txt = np.array(splitted_txt)
    print("splitted text 0 = ", splitted_txt[0])
    print("splitted text 1 = ", splitted_txt[1])
    print("splitted text 2 = ", splitted_txt[2])
    for j in range(2):
      splitted_txt[j] = splitted_txt[j].strip()

    splitted_txt[0] = splitted_txt[0] + '.'
    aggregates.append(splitted_txt[0])
    lexico_semantic_relations.append(splitted_txt[1])

    #csv_data[f'{data_type}'] = aggregates
  csv_data['aggregates'] = aggregates
  #csv_data['masked_target'] = masked_targets
  csv_data['lexico_semantic_relation'] = lexico_semantic_relations
  print("CSV = ", csv_data)
  csv = csv_data.to_csv(f"{output_name}.csv", encoding="utf-8-sig")

  return csv


if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.cuda.current_device()
else:
    print("Will work on CPU.")

filename = "input_raw.txt"
file = open(filename, 'rt')
text = file.readlines()
file.close()


text = np.array(text)
print("TEXT SHAPE = ", text.shape)

##  CREATE CSV FROM TXT FILE, WITH 3 COLUMNS : MASKED SENTENCE, TARGET WORD, LEXICO-SEMANTIC RELATION
#csv = pd.read_csv("input_masked_training.csv", encoding='unicode_escape', names=['masked_sentence', 'masked_target', 'lexico_semantic_relation'])
csv_df = pd.DataFrame(columns=['masked_sentence', 'masked_target', 'lexico_semantic_relation'])

masked_sentences = []
masked_targets = []
lexico_semantic_relations = []

for i in range(text.shape[0]):
  if text[i] == '\n':
    continue

  splitted_sentence = text[i].split(";")
  splitted_sentence = np.array(splitted_sentence)
  print("splitted sentence 0 = ", splitted_sentence[0])
  print("splitted sentence 1 = ", splitted_sentence[1])
  print("splitted sentence 2 = ", splitted_sentence[2])

  for j in range(2):
    splitted_sentence[j] = splitted_sentence[j].strip()


  splitted_sentence[0] = splitted_sentence[0] + '.'
  masked_sentences.append(splitted_sentence[0])
  masked_targets.append(splitted_sentence[1])
  lexico_semantic_relations.append(splitted_sentence[2])

csv_df['masked_sentence'] = masked_sentences
csv_df['masked_target'] = masked_targets
csv_df['lexico_semantic_relation'] = lexico_semantic_relations
print("CSV = ", csv_df)
### ADD JeuxDeMots DATA for FINETUNING

"""
JDM DATA: (extracted via: http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=2 )
"""

# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased',
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']

modelname = 'flaubert/flaubert_large_cased' ### take small case only for debugging

print("MODEL NAME = ", modelname)

tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForMaskedLM.from_pretrained(modelname)

top_1_token = []
top_2_token = []
top_3_token = []
top_4_token = []
top_5_token = []
top_6_token = []
top_7_token = []
top_8_token = []
top_9_token = []
top_10_token = []

txt_data = pd.DataFrame(columns=["samples"])
for i in range(csv_df.shape[0]):
  csv_df.loc[i][0] = csv_df.loc[i][0].replace("<mask>", f"{tokenizer.mask_token}")
  print("\n JDM MASKED SENTENCE = ", csv_df.loc[i][0])
  print("\n JDM MASKED TARGET = ", csv_df.loc[i][1], "\n ")
  #print("\n JDM LEXICO-SEMANTIC RELATION = ", csv_df.loc[i][2])
  #txt_data[samples] = csv_df.loc[i][0])
  #txt_data.append(csv_df.loc[i][1])
  sequence = (
      csv_df.loc[i][0]
  )

  inputs = tokenizer(sequence, return_tensors="pt")
  mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
  token_logits = model(**inputs).logits
  mask_token_logits = token_logits[0, mask_token_index, :]
  top_10_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()
  k = 0
  for token in top_10_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
    #txt_data.append(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
    if k == 0:
      top_1_token.append(tokenizer.decode([token]))
    elif k == 1 :
      top_2_token.append(tokenizer.decode([token]))
    elif k == 2:
      top_3_token.append(tokenizer.decode([token]))
    elif k == 3:
      top_4_token.append(tokenizer.decode([token]))
    elif k == 4:
      top_5_token.append(tokenizer.decode([token]))
    elif k == 5:
      top_6_token.append(tokenizer.decode([token]))
    elif k == 6:
      top_7_token.append(tokenizer.decode([token]))
    elif k == 7:
      top_8_token.append(tokenizer.decode([token]))
    elif k == 8:
      top_9_token.append(tokenizer.decode([token]))
    elif k == 9:
      top_10_token.append(tokenizer.decode([token]))
    else:
      print("error, you should not be here, token = ", token, " : ", tokenizer.decode([token]))

    k+=1
top_tokens = [top_1_token, top_2_token, top_3_token, top_4_token, top_5_token, top_6_token, top_7_token, top_8_token, top_9_token, top_10_token]
i=1
for top_i_token in top_tokens:
  csv_df[f'mask_pred_{i}'] = top_i_token
  i+=1


print("CSV DATAFRAME AFTER TOP 5 TOKEN PREDS = ", csv_df)
#numpy_array = csv_df.to_numpy()
#np.savetxt("results_flauBERT_JDM.txt", numpy_array, fmt = "%s")

csv_df.to_csv('results_flaubert_jdm.csv', encoding='utf-8-sig')
