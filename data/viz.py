import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/freMTPL2freq.csv')
print(df.head())
df.info()
df.describe()
print("finished")