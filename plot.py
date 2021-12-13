import matplotlib.pyplot as plt
import pandas as pd

FILE = 'analysis.csv'

results = pd.read_csv(FILE, names=['name', 'size', 'correct_labels', 'definite_answers', 'accuracy'], index_col=False)
print(results)
size = [int(r.split('-')[-1]) for r in results['name']]
results['em_size'] = size
print(size)
ax = results.plot(x='name', y='em_size', kind='bar', width=0.2, color='purple', bottom=0)
ax.set_title('Accuracy: embedding size vs corpus type')
ax.set_ylabel('Embedding size')
ax = results.plot(x='name', y='accuracy', ax=ax, secondary_y=True)
ax.set_ylabel('Accuracy')
ax.figure.autofmt_xdate()
plt.show()
ax.figure.savefig('plot')