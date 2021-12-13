from gensim.downloader import load
import numpy as np
import pandas as pd

models = [
    'word2vec-google-news-300',
    'glove-wiki-gigaword-100',
    'glove-twitter-100',
    'glove-wiki-gigaword-50',
    'glove-wiki-gigaword-300',
]
FILE = 'synonyms.csv'
for model_name in models:
  print(model_name)
  model = load(model_name)
  df = pd.read_csv(FILE)
  for idx, row in df.iterrows():
    # print(row, row['question'])
    for i in range(0, 4):
      # print(row[i])
      try:
        df.loc[idx, str(i) + '_score'] = model.similarity(row['question'], row[str(i)])
      except:
        df.loc[idx, str(i) + '_score'] = np.nan
    max_val = df.iloc[idx].filter(like='score').astype('float64').idxmax()
    if type(max_val) == str:
      answer = df.columns.get_loc(max_val)
      df.loc[idx, 'guess_word'] = df.iloc[idx, answer - 4]
      if df.loc[idx, 'guess_word'] == df.loc[idx, 'answer']:
        df.loc[idx, 'label'] = 'correct'
      else:
        df.loc[idx, 'label'] = 'incorrect'
    else:
      df.loc[idx, 'label'] = 'guess'
  df[['question', 'answer', 'guess_word', 'label']].to_csv(model_name + '-details.csv', index=False)
  counts = df['label'].value_counts()
  print(counts)
  correct = counts['correct'] if 'correct' in counts.index else 0
  incorrect = counts['incorrect'] if 'incorrect' in counts.index else 0
  guess = counts['guess'] if 'guess' in counts.index else 0
  with open('analysis.csv', 'a') as f:
    f.write(f'{model_name},{len(model)},{correct},{correct+incorrect},{correct/(correct+incorrect)},\n')
  del model
