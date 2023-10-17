# Loaded variable 'df' from URI: /Users/neelanpather/dev/neeland/local-glm-net/data/freMTPL2freq.csv
import pandas as pd
df = pd.read_csv(r'/Users/neelanpather/dev/neeland/local-glm-net/data/freMTPL2freq.csv')

# Change column type to category for column: 'Area'
df = df.astype({'Area': 'category'})

# Change column type to category for column: 'VehBrand'
df = df.astype({'VehBrand': 'category'})

# Change column type to category for column: 'VehGas'
df = df.astype({'VehGas': 'category'})

# Change column type to category for column: 'Region'
df = df.astype({'Region': 'category'})