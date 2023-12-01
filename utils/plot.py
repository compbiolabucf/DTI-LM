import numpy as np 
import matplotlib.pyplot as plt

auc = [0.76, 0.841,	0.891,	0.928,	0.938,	0.943]
auprc = [0.393, 0.632,	0.776,	0.89,	0.926,	0.95]


plt.rcParams['legend.edgecolor'] = 'black'

x = np.arange(6)
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, auc, width, label='AUROC')
rects2 = ax.bar(x + width/2, auprc, width, label='AUPRC')
ax.set_ylabel('Scores')
ax.set_xlabel('Number of leaked samples')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()
plt.savefig('bar1.pdf')

