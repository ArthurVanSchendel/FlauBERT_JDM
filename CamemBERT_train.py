import torch
import transformers
import numpy as np
import pandas as pd
import math
from urllib.request import urlopen
from bs4 import BeautifulSoup

from transformers import CamembertForMaskedLM, CamembertTokenizer, AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer

from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import logging

from datasets import load_dataset

logger = logging.getLogger()

DATA_PATH = Path('./data/')
LOG_PATH = Path('./logs/')
MODEL_PATH = Path('./model/')
LABEL_PATH = Path('./labels/')

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
    for j in range(len(sentences)):
      indexes.append(j)
    final_txt = pd.DataFrame(columns=['sentences', 'lexico_sem_relation'], index=indexes)
    #final_txt = pd.DataFrame(columns=['sentences'])
    for i in range(len(sentences)):
      text = str(sentences[i]).replace('<sen>', '')
      text = text.replace('</sen>', '.')
      lex_sem = str(lex_sem_relations[i]).replace('<ch>', '')
      lex_sem = lex_sem.replace('</ch>', '')
      lex_sem = lex_sem.replace('&gt;', '>')

      final_txt['sentences'].loc[i] = text
      final_txt['lexico_sem_relation'].loc[i] = lex_sem
    return final_txt


def create_dataset(nb_calls, url, type_dataset):
  indexes = []
  final_dataset = pd.DataFrame(columns=['sentences', 'lexico_sem_relation'])
  #tmp_dataset = []
  for i in range(nb_calls):
    strip = extract_txt_from_html(url)
    final_dataset = pd.concat([final_dataset, strip])
  final_dataset.to_csv(f"aggregate_{type_dataset}.txt", encoding="utf-8-sig", columns=['sentences'])
  return final_dataset

url_data = "http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=10"
modelname = 'camembert-base'

txt_train = create_dataset(3, url_data, "train")
txt_valid = create_dataset(2, url_data, "valid")

#tokenizer = CamembertTokenizer.from_pretrained(modelname, do_lower_case=False)
tokenizer = AutoTokenizer.from_pretrained(modelname, padding=True, truncation=True)
model = AutoModelForMaskedLM.from_pretrained(modelname)
