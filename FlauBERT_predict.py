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

def extract_txt_from_html(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    soup.prettify()
    indexes = []
    sentences = soup.find_all('sen')
    lex_sem_relations = soup.find_all('ch')
    masks = soup.find_all('')
    for j in range(len(sentences)):
      indexes.append(j)
    final_txt = pd.DataFrame(columns=['masked_sentences', 'masked_target', 'lexico_sem_relation'], index=indexes)
    #final_txt = pd.DataFrame(columns=['sentences'])
    for i in range(len(sentences)):
      tmp_txt = pd.DataFrame(columns=['masked_sentences', 'masked_target', 'lexico_sem_relation'], index=indexes)
      text = str(sentences[i]).replace('<sen>', '')
      text = text.replace('</sen>', '.')
      lex_sem = str(lex_sem_relations[i]).replace('<ch>', '')
      lex_sem = lex_sem.replace('</ch>', '')
      lex_sem = lex_sem.replace('&gt;', '>')

      final_txt['masked_sentences'].loc[i] = text
      final_txt['masked_target'].loc[i] = masks
      final_txt['lexico_sem_relation'].loc[i] = lex_sem
    return final_txt

def create_dataset(nb_calls, url, type_dataset):
  indexes = []
  final_dataset = pd.DataFrame(columns=['masked_sentences', 'lexico_sem_relation'])
  #tmp_dataset = []
  for i in range(nb_calls):
    strip = extract_txt_from_html(url)
    final_dataset = pd.concat([final_dataset, strip])
  final_dataset.to_csv(f"aggregate_{type_dataset}.txt", encoding="utf-8-sig", columns=['masked_sentences'])
  return final_dataset


"""
JDM DATA:(extracted via: http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=5
"""

url_data = 'http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=5&mask=1'

sample_data = create_dataset(2, url_data, "test")
print("SAMPLE DATA = ", sample_data)
dataset = load_dataset('text', data_files={'test': 'aggregate_test.txt'})
print("DATASET = ", dataset)
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


for i in range(csv_df.shape[0]):
  csv_df.loc[i][0] = csv_df.loc[i][0].replace("<mask>", f"{tokenizer.mask_token}")
  print("\n JDM MASKED SENTENCE = ", csv_df.loc[i][0])
  print("\n JDM MASKED TARGET = ", csv_df.loc[i][1], "\n ")
  #print("\n JDM LEXICO-SEMANTIC RELATION = ", csv_df.loc[i][2])

  sequence = (
      csv_df.loc[i][0]
  )

  inputs = tokenizer(sequence, return_tensors="pt")
  mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
  token_logits = model(**inputs).logits
  mask_token_logits = token_logits[0, mask_token_index, :]
  top_10_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()
  k = 0
  for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
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

print("CSV DATAFRAME AFTER TOP 10 TOKEN PREDS = ", csv_df)
csv_df.to_csv('results_FlauBERT_JDM.csv', encoding='utf-8-sig')

####  CREATE A TRAIN SET AND A VALIDATION SET/ input_raw_tr.txt AND input_raw_valid.txt
## THE SETS SHOULD NOT BE MASKED YET, IT WILL BE HANDLED WITHIN BERT (SEE BELOW)
