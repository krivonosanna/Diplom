import numpy as np
import pandas as pd


class Trainer(object):
    r'''Base class for all trainer.'''

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.corr = self.__corr_pirson__()

    def __corr_pirson__(self):
        self.data = self.data.sort_values(by='Дата')
        country = self.data['Страна'].unique()
        n = len(self.data[self.data['Страна'] == country[0]]['Заражений за день'])
        corr = np.eye(len(country))
        for i in range(len(country)):
            for j in range(i + 1, len(country)):
                si = self.data[self.data['Страна'] == country[i]]['Заражений за день']
                sj = self.data[self.data['Страна'] == country[j]]['Заражений за день']
                p = []
                for l in range(21):
                    if np.std(si[:n - l]) != 0 and np.std(sj[l:]) != 0:
                        p.append(np.corrcoef(si[:n - l], sj[l:])[0, 1])
                    if np.std(sj[:n - l]) != 0 and np.std(si[l:]) != 0:
                        p.append(np.corrcoef(si[l:], sj[:n - l])[0, 1])
                if p == []:
                    corr[i, j] = 0
                    corr[j, i] = 0
                else:
                    corr[i, j] = np.max(p)
                    corr[j, i] = np.max(p)
        return corr

    def fit_predict(self):
        labels = self.model.fit_predict(1 - self.corr)
        clasters = []
        country = self.data['Страна'].unique()
        for i in pd.Series(labels).unique():
            clasters.append(np.array(country)[labels == i])
        clasters = sorted(clasters, key=len)[::-1]
        return clasters
