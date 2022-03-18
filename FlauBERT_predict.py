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
from transformers.pipelines.base import KeyDataset
from tqdm.auto import tqdm


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
  final_dataset = pd.DataFrame(columns=['masked_sentences', 'masked_target', 'lexico_sem_relation'])
  #tmp_dataset = []
  for i in range(nb_calls):
    strip = extract_txt_from_html(url)
    final_dataset = pd.concat([final_dataset, strip], ignore_index=True)
  final_dataset.to_csv(f"aggregate_{type_dataset}.txt", encoding="utf-8-sig", columns=['masked_sentences', 'masked_target', 'lexico_sem_relation'])
  return final_dataset

def dataset_to_dataframe(dataset, indexes):
    dataframe = pd.DataFrame(columns=['sentence', 'masked_target', 'lexico_sem_relation'])
    sentences = []
    masks = []
    lexico_sem_relations = []
    for i in range(np.array(dataset['test']['text']).shape[0]):
        split = dataset['test']['text'][i].split(",")
        sentences.append(split[1])
        masks.append(split[2])
        lexico_sem_relations.append(split[3])
    dataframe['sentence'] = sentences
    dataframe['masked_target'] = masks
    dataframe['lexico_sem_relation'] = lexico_sem_relations

    return dataframe
"""
JDM DATA:(extracted via: http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=33&verbose=0&iter=1
"""

#url_data = 'http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=100&verbose=0&iter=1&mask=1'

url_data = 'http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=100&verbose=0&iter=1&mask=1'

#sample_data = create_dataset(1, url_data, "test")
#print("SAMPLE DATA = ", sample_data)
dataset = load_dataset('text', data_files={'test': 'aggregate_baseline.txt'})


print("DATASET = ", dataset)
print("DATASET['test']['text'] = ", dataset['test']['text'])
# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased',
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']

modelname = 'flaubert/flaubert_large_cased' ### take small case only for debugging

print("MODEL NAME = ", modelname)

tokenizer = AutoTokenizer.from_pretrained(modelname)
#model = AutoModelForMaskedLM.from_pretrained(modelname, return_dict=True).to(device)
model = AutoModelForMaskedLM.from_pretrained("retrained_flaubert/flaubert_large_cased", return_dict=True).to(device)
#model = AutoModelForMaskedLM.from_pretrained("flaubert/flaubert_large_cased-finetuned-JDM_text/checkpoint-500/config.json", return_dict=True, from_tf=True).to(device)

top_answers_nb = 10
lex_rel = []
indexes=[]



for i in range(dataset['test'].shape[0]-1):
    indexes.append(i)
print("indexes = ", indexes)

dataframe = dataset_to_dataframe(dataset, indexes)

new_dataset = []
info = pd.DataFrame(columns=['score', 'sequence', 'mask'])

for i in range(dataset['test'].shape[0]):
    if "<mask>" not in dataset['test']['text'][i]:
        continue
    split = dataset['test']['text'][i].split(",")
    split[1] = split[1].capitalize()
    split[1] = split[1].replace("<mask>", f"{tokenizer.mask_token}")
    dataset['test']['text'][i] = split[1]
    new_dataset.append(split[1])

print("new_dataset = ", new_dataset)
print("new_dataset.shape = ", np.array(new_dataset).shape)
print('info = ', info)
#nlp_fill = pipeline('fill-mask', model='flaubert/flaubert_large_cased', top_k=top_answers_nb, device=0)
nlp_fill = pipeline('fill-mask', model='retrained_flaubert/flaubert_large_cased', tokenizer=tokenizer,top_k=top_answers_nb, device=0)
i=0

info = pd.DataFrame(columns=["score", "sequence", "mask"], index=indexes)
print("new_dataset = ", new_dataset)
for out in tqdm(nlp_fill(new_dataset)):
    scores = []
    sequences = []
    masks = []
    for j in range(top_answers_nb):
        scores.append(out[j]['score'])
        sequences.append(out[j]['sequence'])
        masks.append(out[j]['token_str'])

    info['score'].loc[i] = scores
    info['sequence'].loc[i] = sequences
    info['mask'].loc[i] = masks
    i+=1

print("info = ", info)

with open('result_flauBERT_JDM.txt', 'w', encoding='utf-8') as f:
    init_lines = ['///', modelname, f'cycle = {0} - (baseline)', 'train parameters = [learning_rate=3e-5, num_train_epochs=30, per_device_train_batch_size=8, logging_steps=1, weight_decay=0.01,] ','/// \n']
    f.write('\n'.join(init_lines))
    f.write('\n')
    for i in range(info.shape[0]):
        f.write(f"\n ENTRY: <sen> {dataframe.loc[i+1][0]} </sen>    ;    <m> {dataframe.loc[i+1][1]} </m>    ;    <ch> {dataframe.loc[i+1][2]} </ch>    ; \n")
        for j in range(top_answers_nb):
            #print("i = ", i)
            #print("j = ", j)
            masks = info['mask'].loc[i][j]
            scores = info['score'].loc[i][j]
            f.write(f'\n ANSWER {j}:    {masks}    ;    {scores}    ; \n')
        f.write('\n')




#print("CSV DATAFRAME AFTER TOP 10 TOKEN PREDS = ", sample_data)
#sample_data.to_csv('results_FlauBERT_JDM.txt', encoding='utf-8')

####  CREATE A TRAIN SET AND A VALIDATION SET/ input_raw_tr.txt AND input_raw_valid.txt
## THE SETS SHOULD NOT BE MASKED YET, IT WILL BE HANDLED WITHIN BERT (SEE BELOW)
