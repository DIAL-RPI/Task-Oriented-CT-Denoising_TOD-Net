import os
import sys
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

primary_dir = 'Z:/proj/moseg/models'
metric_names = ['DSC','ASD']
class_names = ['Liver', 'Kidney', 'Spleen']
modelpaths = [
    ['{}/model_20200922225715'.format(primary_dir),'trial 1'],
    ['{}/model_20200923205915'.format(primary_dir),'trial 2'],
    ['{}/model_20200924011346'.format(primary_dir),'trial 3']
]

data = {}
plt.figure()
for [modelpath, modelname] in modelpaths:
    d = np.loadtxt('{}/loss.txt'.format(modelpath))
    data[modelpath] = d
    plt.plot(d[:,8],label=modelname)
plt.xlim((0, 100))
plt.ylim((-0.05, 0.9))
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()
plt.grid(True)
plt.show()