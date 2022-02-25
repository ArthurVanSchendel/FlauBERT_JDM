import torch
import transformers
import numpy as np
import pandas as pd
import csv

from transformers import XLMTokenizer, XLMWithLMHeadModel, pipeline
from transformers import FlaubertModel, FlaubertTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer

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
######

#  HERE ADD CONVERTION FROM DATARFAME TO CSV (FOR TRAINING / TESTING  ## )

#csv = csv_df.to_csv("input_mask.csv")

#print("REAL CSVVVV = ", csv)

########

### ADD JeuxDeMots DATA for FINETUNING

"""
JDM DATA: (extracted via: http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=4 )
"""

#jdm_masked_data = pd.read_csv('input_masked.csv', encoding='unicode_escape', names=['masked_sentence', 'masked_target', 'lexico_semantic_relation'])

#masked_sentences = jdm_masked_data.masked_sentence.to_list()
#masked_targets = jdm_masked_data.masked_target.to_list()


# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased',
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']

modelname = 'flaubert/flaubert_large_cased' ### take small case only for debugging

print("MODEL NAME = ", modelname)

tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForMaskedLM.from_pretrained(modelname)

for i in range(csv_df.shape[0]):
  csv_df.loc[i][0] = csv_df.loc[i][0].replace("<mask>", f"{tokenizer.mask_token}")
  print("\n JDM MASKED DATA LOC [i] [0] = ", csv_df.loc[i][0])
  print("\n JDM MASKED TARGET = ", csv_df.loc[i][1], "\n ")

  sequence = (
      csv_df.loc[i][0]
  )

  inputs = tokenizer(sequence, return_tensors="pt")
  mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
  token_logits = model(**inputs).logits
  mask_token_logits = token_logits[0, mask_token_index, :]
  top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

  for token in top_5_tokens:
      print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))


#classifier = pipeline("fill-mask", model=modelname, tokenizer=tokenizer, topk=10)
#print(classifier(f"La capitale de la France est {tokenizer.mask_token}."))
