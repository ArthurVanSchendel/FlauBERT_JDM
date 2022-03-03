import torch
import transformers
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd4
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from urllib.request import urlopen

from datasets import load_dataset

if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.cuda.current_device()
else:
    print("Will work on CPU.")


def extract_txt_from_html(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    final_txt = []
    soup.prettify()
    sentences = soup.find_all('sen')
    print("SENTENCES = ", sentences)
    print("len sentences = ", len(sentences))
    for i in range(len(sentences)):
      text = str(sentences[i]).replace("<sen>", '')
      text = text.replace("</sen>", '.')
      final_text.append(text)
    print("sentences after replace = ", final_text)
    return final_text

url = "http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=4&verbose=0&iter=2"

content = extract_txt_from_html(url)

print("CONTENT = ", content)
print("LEN CONTENT = ", len(content))
