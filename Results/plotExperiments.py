import time
import DataLogging as dl
import numpy as np

Struct = dl.Struct

cmpForward = dl.loadAllRuns('data/Compliance/Forward')
cmpReverse = dl.loadAllRuns('data/Compliance/Reverse')
erpForward = dl.loadAllRuns('data/ExperienceReplay/Forward')
erpReverse = dl.loadAllRuns('data/ExperienceReplay/Reverse')
er = erpForward + erpReverse
cmp = cmpForward + cmpReverse

data = er + cmp
print(list(len(e.dataNames) for e in er))
er  = list(filter(lambda run : len(run.dataNames) == 12, data))
cmp = list(filter(lambda run : len(run.dataNames) == 10, data))

def expToScore(exp):
    pos = exp.extractData('currentpos')
    return np.nanmean([dl.mag(pos[177+i]-pos[i]) for i in range(len(pos)-177)])

import pickle
with open('pexperiments01.snake', 'wb') as file:
    pickle.dump(er, file)

scoresR = 1 / np.array(list(map(expToScore, er))[:10])
scoresC = 1 / np.array(list(map(expToScore, cmp))[:10])

Rscore = np.mean(scoresR)
Cscore = np.mean(scoresC)

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
##font = {'family' : 'DejaVu Sans',
##        'weight' : 'bold',
#font = {'family' : 'Times New Roman',
#        'size'   : 30}
#matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 40

'''
plt.plot(scoresR)
plt.plot(scoresC)
plt.plot([Rscore for score in scoresR])
plt.plot([Cscore for score in scoresC])
print(np.mean(scoresR), np.mean(scoresC))
print(np.std(scoresR), np.std(scoresC))

plt.show()
'''

fig, ax = plt.subplots(figsize=(16,7))
ind = np.arange(10)
width = 0.85
ebar = {'capsize': 4,
        'mew': 4,
        'linewidth':4}
stdDev1 = ax.barh([0], [Cscore], width, color = 'r', xerr=np.std(scoresC), error_kw = ebar)
rects1 = ax.barh(ind+1, scoresC, width, color='gray')
stdDev2 = ax.barh([12], [Rscore], width, color = 'r', xerr = np.std(scoresR), error_kw = ebar)
rects2 = ax.barh(ind+13, scoresR, width, color = 'black')

ax.legend((rects2[0], rects1[0]), ('Learned Policy', 'Compliance'))
ax.set_xlabel('Time to Traverse [Cycles/m] (Lower is better)')
yticks = np.arange(max(ind)+14)
yticks[11] = 0
ax.set_yticks(yticks)

ylabels = ['' for t in yticks]
ylabels[0] = 'Average'
ylabels[12] = 'Average'
ax.set_yticklabels(ylabels)

'''
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        width  = rect.get_width()
        ax.text(rect.get_y() + height/2., 1.05*width,
                '%f' % round(width,3),
                ha='right', va='center')
#autolabel(rects1)
#autolabel(rects2)
'''

plt.show()
fig.savefig('results.eps')
