from transformers.utils.dummy_flax_objects import FlaxAutoModelForSeq2SeqLM
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

#csv_file = pd.DataFrame(list())
#csv_file.to_csv('input_masked_training.csv')

text = np.array(text)
print("TEXT SHAPE = ", text.shape)
#print("TEXT = \n ", text)

##  CREATE CSV FROM TXT FILE, WITH 3 COLUMNS : MASKED SENTENCE, TARGET WORD, LEXICO-SEMANTIC RELATION
#csv = pd.read_csv("input_masked_training.csv", encoding='unicode_escape', names=['masked_sentence', 'masked_target', 'lexico_semantic_relation'])

masked_sentences = []
masked_targets = []
lexico_semantic_relations = []

for i in range(5):  #text.shape[0]
  if text[i] == "\n":
    break

  splitted_sentence = text[i].split(";")

  print("SPLITTED SENTENCE BEFORE STRIP = ", splitted_sentence)
  #print("SPLITTED SENTENCE BEFORE STRIP SHAPE = ", splitted_sentence.shape)
  #for j in range(3):
  splitted_sentence[0] = splitted_sentence[0].strip()
  splitted_sentence[1] = splitted_sentence[1].strip()
  splitted_sentence[2] = splitted_sentence[2].strip()

  print("SPLITTED SENTENCE AFTER STRIP = ", splitted_sentence)
  splitted_sentence[0] = splitted_sentence[0] + '.'
  print("SPLITTED SENTENCE WITH DOT = ", splitted_sentence)
  #print("SPLITTED SENTENCE AFTER STRIP SHAPE = ", splitted_sentence.shape)
  #print("SPLITTED SENTENCE = ", splitted_sentence)
  #print("SPLITTED SENTENCE SHAPE = ", splitted_sentence.shape)
  masked_sentences.append(splitted_sentence[0])
  masked_targets.append(splitted_sentence[1])
  lexico_semantic_relations.append(splitted_sentence[2])

  #d = {"masked_sentence" : [splitted_sentence[0]],
  #     "masked_target" : [splitted_sentence[1]],
  #     "lexico_semantic_relation" : [splitted_sentence[2]]}
  #print("D EQUAL TO = ", d)

  #masked_sentence_dict = {"masked_sentence" : splitted_sentence[0]}
  #masked_target_dict = {"masked target" : splitted_sentence[1]}
  #lexico_semantic_relation_dict = {"lexico_semantic_relation" : splitted_sentence[2]}


  #masked_sentence_df = pd.DataFrame(masked_sentence_dict, index=[0, 1, 2])
  #masked_target_df = pd.DataFrame(masked_target_dict, index=[0, 1, 2])
  #lexico_semantic_relation_df = pd.DataFrame(lexico_semantic_relation_dict, index=[0, 1, 2])

  #csv.append(masked_sentence_df)
  #csv.append(masked_target_df)
  #csv.append(lexico_semantic_relation_df)
masked_sentences_df = pd.DataFrame(masked_sentences, columns=['masked_sentence'])
masked_targets_df = pd.DataFrame(masked_targets, columns=['masked_target'])
lexico_semantic_relations_df = pd.DataFrame(lexico_semantic_relations, columns=['lexico_semantic_relation'])

print("MASKED SENTENCE DF = ", masked_sentences_df)
print("MASKED TARGETS DF = ", masked_targets_df)
print("LEXICO SEMANTIC RELATION = ", lexico_semantic_relations_df)

#print("CSV FILE = ", csv_file_df)
#print("CSV FILE SHAPE = ", csv_file_df.shape)
#print("CSV FILE ELEMENT 0 = ", csv_file_df.loc[0])
#print("CSV FILE ELEMENT 1 = ", csv_file_df.loc[1])
#print("CSV FILE ELEMENT 2 = ", csv_file_df.loc[2])

######

#  HERE ADD CONVERTION FROM DATARFAME TO CSV (FOR TRAINING / TESTING  ## )

########

test = pd.read_csv('input_training_mask.csv', encoding='unicode_escape', names=['masked_sentence', 'masked_target', 'lexico_semantic_relation'])
print("TEST TEST TEST = ", test)
print("TEST TEST TEST LENGTH = ", len(test))

print("TEST = ", test.masked_sentence)
print("TEST LENGTH = ", len(test.masked_sentence))
#print("CSV = \n ", csv)
### ADD JeuxDeMots DATA for FINETUNING

"""
JDM DATA: (extracted via: http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=20&verbose=0&iter=2 )
"""

jdm_masked_data = pd.read_csv('input_masked.csv', encoding='unicode_escape', names=['masked_sentence', 'masked_target', 'lexico_semantic_relation'])

masked_sentences = jdm_masked_data.masked_sentence.to_list()
masked_targets = jdm_masked_data.masked_target.to_list()
print("####################################################")
print("MASKED DATA FROM JDM = \n", jdm_masked_data)
print("SHAPE = ", jdm_masked_data.shape)
print("JDM MASKED SENTENCES = ", masked_sentences)
print("JDM MASKED TARGET = ", masked_targets)


# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased',
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']
modelname = 'flaubert/flaubert_large_cased' ### take small case only for debugging

print("MODEL NAME = ", modelname)

tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForMaskedLM.from_pretrained(modelname)

for i in range(jdm_masked_data.shape[0]):
  jdm_masked_data.loc[i][0] = jdm_masked_data.loc[i][0].replace("<mask>", f"{tokenizer.mask_token}")
  print("\n JDM MASKED DATA LOC [i] [0] = ", jdm_masked_data.loc[i][0])
  print("\n JDM MASKED TARGET = ", jdm_masked_data.loc[i][1], "\n ")

  sequence = (
      jdm_masked_data.loc[i][0]
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
