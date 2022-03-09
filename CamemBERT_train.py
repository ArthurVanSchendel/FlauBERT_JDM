import torch
import transformers
import numpy as np
import pandas as pd
import math
from urllib.request import urlopen
from bs4 import BeautifulSoup

from transformers.modeling_camembert import CamembertForMaskedLM
from transformers.tokenization_camembert import CamembertTokenizer
from fast_bert.data_lm import BertLMDataBunch
from fast_bert.learner_lm import BertLMLearner
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
      tmp_txt = pd.DataFrame(columns=['sentences', 'lexico_sem_relation'], index=indexes)
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

txt_train = create_dataset(3, url_data, "train")
txt_valid = create_dataset(2, url_data, "valid")

databunch_lm = BertLMDataBunch.from_raw_corpus(

                    text_list= txt_train,
                    tokenizer= 'camembert-base',
                    batch_size_per_gpu= 8,
                    max_seq_length= 512,
                    multi_gpu= False,
                    model_type= 'camembert-base',
                    logger=logger)

learner_lm = BertLMLearner.from_pretrained_model(
                            dataBunch=databunch_lm,
                            pretrained_path='camembert-base',
                            output_dir=MODEL_PATH,
                            metrics=[],
                            device=device,
                            logger=logger,
                            multi_gpu=False,
                            logging_steps=50,
                            fp16_opt_level="O2")

lm_learner.fit(epochs=10,
            lr=2e-5,
            validate=True,
            schedule_type="warmup_cosine",
            optimizer_type="adamw")

lm_learner.validate()
lm_learner.save_model()
