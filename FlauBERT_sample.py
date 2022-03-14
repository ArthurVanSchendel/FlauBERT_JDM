import transformers
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datasets import load_dataset
import pandas as pd


def extract_txt_from_html(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    soup.prettify()
    indexes = []
    sentences = soup.find_all('sen')
    print("sentences = ", sentences)
    lex_sem_relations = soup.find_all('ch')
    print("lexico relations = ", lex_sem_relations)
    masked_targets = soup.find_all('m')
    print("masked targets = ", masked_targets)
    for j in range(len(sentences)):
      indexes.append(j)
    final_txt = pd.DataFrame(columns=['sentences', 'masks','lexico_sem_relations'], index=indexes)
    #final_txt = pd.DataFrame(columns=['sentences'])
    for i in range(len(sentences)):
      text = str(sentences[i]).replace('<sen>', '')
      masks = str(masked_targets[i]).replace('<m>', '')
      lex_sem = str(lex_sem_relations[i]).replace('<ch>', '')

      text = text.replace('</sen>', '.')
      text = text.replace('&lt;', '<')
      text = text.replace('&gt;', '>')

      masks = masks.replace('</m>', '')
      masks = masks.replace('&gt;', '>')
      masks = masks.replace('&lt;', '<')

      lex_sem = lex_sem.replace('</ch>', '')
      lex_sem = lex_sem.replace('&gt;', '>')
      text = text.capitalize()
      final_txt['sentences'].loc[i] = text
      final_txt['masks'].loc[i] = masks
      final_txt['lexico_sem_relations'].loc[i] = lex_sem
    return final_txt


def create_dataset(nb_calls, url, type_dataset):
  indexes = []
  final_dataset = pd.DataFrame(columns=['sentences', 'masks', 'lexico_sem_relations'])
  #tmp_dataset = []
  for i in range(nb_calls):
    strip = extract_txt_from_html(url)
    final_dataset = pd.concat([final_dataset, strip], ignore_index=True)
  final_dataset.to_csv(f"aggregate_{type_dataset}.txt", encoding="utf-8-sig", columns=['sentences', 'masks', 'lexico_sem_relations'])
  return final_dataset

url_data = "http://www.jeuxdemots.org/intern_interpretor.php?chunks-display=1&chunk=4&verbose=0&iter=5&mask=1"

txt_baseline = create_dataset(7, url_data, "baseline")
