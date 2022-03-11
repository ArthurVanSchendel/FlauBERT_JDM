import torch
import transformers
import numpy as np
import pandas as pd
import csv
import math

from urllib.request import urlopen
from bs4 import BeautifulSoup

from transformers import XLMTokenizer, XLMWithLMHeadModel
from transformers import FlaubertModel, FlaubertTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

from torch.nn import functional as F

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
    masked_targets = soup.find_all('m')
    for j in range(len(sentences)):
      indexes.append(j)
    final_txt = pd.DataFrame(columns=['masked_sentences', 'masked_target', 'lexico_sem_relation'], index=indexes)
    #final_txt = pd.DataFrame(columns=['sentences'])
    for i in range(len(sentences)):
      text = str(sentences[i]).replace('<sen>', '')
      text = text.replace('</sen>', '.')
      text = text.replace('&gt;', '>')
      text = text.replace('&lt;', '<')

      masks = str(masked_targets[i]).replace('<m>', '')
      masks = masks.replace('</m>', '')
      masks = masks.replace('&gt;', '>')
      masks = masks.replace('&lt;', '<')

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
    final_dataset = pd.concat([final_dataset, strip], ignore_index=True)
  final_dataset.to_csv(f"aggregate_{type_dataset}.txt", encoding="utf-8-sig", columns=['masked_sentences'])
  return final_dataset


"""
JDM DATA:(extracted via: http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=2
"""

url_data = 'http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=4&verbose=0&iter=1&mask=1'

#sample_data = create_dataset(8, url_data, "test")
print("SAMPLE DATA = ", sample_data)
dataset = load_dataset('text', data_files={'test': 'aggregate_baseline.txt'})
print("DATASET = ", dataset)
# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased',
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']

modelname = 'flaubert/flaubert_large_cased' ### take small case only for debugging

print("MODEL NAME = ", modelname)

tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForMaskedLM.from_pretrained(modelname, return_dict=True)

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

print("shape of sample data = ", sample_data.shape)
top_answers_nb = 10
lex_rel = []




indexes=[]
for i in range(sample_data.shape[0]):
    indexes.append(i)
print("indexes = ", indexes)
info = pd.DataFrame(columns=['score', 'sequence', 'mask'], index=indexes)

for i in range(sample_data.shape[0]):
    score = []
    sen =  []
    mask = []
    sample_data.loc[i][0] = sample_data.loc[i][0].replace("<mask>", f"{tokenizer.mask_token}")
    sample_data.loc[i][0] = sample_data.loc[i][0].capitalize()
    print("\n JDM MASKED SENTENCE = ", sample_data.loc[i][0])
    print("\n JDM MASKED TARGET = ", sample_data.loc[i][2])
    print("\n JDM LEXICO-SEMANTIC RELATION = ", sample_data.loc[i][1], "\n")
    lex_rel.append(sample_data.loc[i][1])


    sequence = (
        sample_data.loc[i][0]
    )
    nlp_fill = pipeline('fill-mask', model="flaubert/flaubert_large_cased", top_k=top_answers_nb)
    output = nlp_fill(sample_data.loc[i][0])


    for j in range(top_answers_nb):
        dict = output[j]
        score.append(dict['score'])
        sen.append(dict['sequence'])
        mask.append(dict['token_str'])

    info['score'].loc[i] = score
    info['sequence'].loc[i] = sen
    info['mask'].loc[i] = mask
    print("INFO = ", info)
    print("output = ", output)
    print("score = ", score)
    print("sen = ", sen)
    print("mask = ", mask)
    print("lex rel = ", lex_rel)

print('info = ', info)
with open('result_flauBERT_JDM.txt', 'w', encoding='utf-8') as f:
    init_lines = ['///', modelname, f'cycle = {0} - (baseline)', 'train parameters = [] ','/// \n']
    f.write('\n'.join(init_lines))
    f.write('\n')
    for i in range(sample_data.shape[0]):
        f.write(f'\n ENTRY: <sen> {sample_data.loc[i][0]} </sen>    ;    <m> {sample_data.loc[i][2]} </m>    ;    <ch> {sample_data.loc[i][1]} </ch>    ; \n')
        for j in range(top_answers_nb):
            masks = info['mask'].loc[i][j]
            scores = info['score'].loc[i][j]
            f.write(f'\n ANSWER {j}:    {masks}    ;    {scores}    ; \n')
        f.write('\n')

  #input = tokenizer.encode_plus(sequence, return_tensors = "pt")
  #labels = tokenizer.encode_plus(sequence, return_tensors = "pt")["input_ids"]
  #mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
  #output = model(**input) # , labels=labels
  #logits = output.logits
  #print("logits = ", logits)
  #softmax_proba = F.softmax(logits, dim = -1)
  #print("softmax proba = ", softmax_proba)
  #mask_word = softmax_proba[0, mask_index, :]
  #print("mask word = ", mask_word)
  #top_10_tokens = torch.topk(mask_word, top_answers_nb, dim = 1)[1][0]
  #print("top 10 tokens = ", top_10_tokens)
  #top_p, top_class = softmax_proba.topk(top_answers_nb, dim=-1)
  #print("top p = ", top_p, " with shape ", top_p.shape)
  #print("top class = ", top_class, " with shape ", top_class.shape)
  #sum_row_top_p = []
  #for j in range(top_p.shape[1]):
#      tmp_sum = 0.0
#      for k in range(top_p.shape[2]):
#          tmp_sum = tmp_sum + top_p[0][j][k].item()
#      sum_row_top_p.append(tmp_sum)
#
 # print("sum row top p = ", sum_row_top_p, "\n")

  #k = 0
  #for token in top_10_tokens:
#    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
#    if k == 0:
#      top_1_token.append(tokenizer.decode([token]))
#    elif k == 1 :
#      top_2_token.append(tokenizer.decode([token]))
#    elif k == 2:
#      top_3_token.append(tokenizer.decode([token]))
#    elif k == 3:
#      top_4_token.append(tokenizer.decode([token]))
#    elif k == 4:
#      top_5_token.append(tokenizer.decode([token]))
#    elif k == 5:
#      top_6_token.append(tokenizer.decode([token]))
#    elif k == 6:
#      top_7_token.append(tokenizer.decode([token]))
#    elif k == 7:
#      top_8_token.append(tokenizer.decode([token]))
#    elif k == 8:
#      top_9_token.append(tokenizer.decode([token]))
#    elif k == 9:
#      top_10_token.append(tokenizer.decode([token]))
#    else:
#      print("error, you should not be here, token = ", token, " : ", tokenizer.decode([token]))

#    k+=1
#top_tokens = [top_1_token, top_2_token, top_3_token, top_4_token, top_5_token, top_6_token, top_7_token, top_8_token, top_9_token, top_10_token]
#i=1
#for top_i_token in top_tokens:
#  sample_data[f'mask_pred_{i}'] = top_i_token
#  i+=1

#print("CSV DATAFRAME AFTER TOP 10 TOKEN PREDS = ", sample_data)
#sample_data.to_csv('results_FlauBERT_JDM.txt', encoding='utf-8')

####  CREATE A TRAIN SET AND A VALIDATION SET/ input_raw_tr.txt AND input_raw_valid.txt
## THE SETS SHOULD NOT BE MASKED YET, IT WILL BE HANDLED WITHIN BERT (SEE BELOW)
