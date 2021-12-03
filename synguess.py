from gensim.downloader import load
from gensim import similarities

print('test')
dataset = load('word2vec-google-news-300')
print(dataset)

similarities.SoftCosineSimilarity()