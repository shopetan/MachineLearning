# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

randn = np.random.randn
ts = Series(randn(1000), index = pd.date_range('2000/1/1', periods = 1000))
ts = ts.cumsum()
ts.plot(style = '<--')
pd.rolling_mean(ts,60).plot(style = '--' ,c = 'r')
pd.rolling_mean(ts,180).plot(style = '--' ,c = 'r')
plt.show()
