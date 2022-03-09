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


def tokenize_function(examples):
    return tokenizer(examples["text"])


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


"""
JDM DATA:(extracted via: http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=5
"""
url_data = 'http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=5'

sample_data = create_dataset(2, url_data, "test")
dataset = load_dataset('text', data_files={'test': 'aggregate_test.txt'})
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
top_9_token = []


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
      top_5_token.append(tokenizer.decode([token]))
    elif k == 6:
      top_5_token.append(tokenizer.decode([token]))
    elif k == 7:
      top_5_token.append(tokenizer.decode([token]))
    elif k == 8:
      top_5_token.append(tokenizer.decode([token]))
    elif k == 9:
      top_5_token.append(tokenizer.decode([token]))
    else:
      print("error, you should not be here, token = ", token, " : ", tokenizer.decode([token]))

    k+=1
top_tokens = [top_1_token, top_2_token, top_3_token, top_4_token, top_5_token]
i=1
for top_i_token in top_tokens:
  csv_df[f'mask_pred_{i}'] = top_i_token
  i+=1

print("CSV DATAFRAME AFTER TOP 5 TOKEN PREDS = ", csv_df)
csv_df.to_csv('results_flaubert_jdm.csv', encoding='utf-8-sig')

####  CREATE A TRAIN SET AND A VALIDATION SET/ input_raw_tr.txt AND input_raw_valid.txt
## THE SETS SHOULD NOT BE MASKED YET, IT WILL BE HANDLED WITHIN BERT (SEE BELOW)

## LOAD THEM HERE ###
datasets = load_dataset("text", data_files={"train": input_raw_tr.txt, "validation": input_raw_valid.txt})
#####################
datasets["train"][10]
model_checkpoint = "flaubert/flaubert_base_cased"

tokenizer_2 = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

model_2 = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    f"{model_name}-finetuned-JDM_text",
    evaluation_strategy = "epoch",
    learning_rate=6e-4,
    weight_decay=0.01,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_2, mlm_probability=0.15)  ###  15% of sentence is masked

trainer = Trainer(
    model=model_2,
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
