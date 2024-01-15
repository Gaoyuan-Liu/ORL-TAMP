import pandas as pd

# Import
df1 = pd.read_csv('training_data_1.csv')
df2 = pd.read_csv('training_data_2.csv')
print(len(df1))
print(len(df2))
# Merge
frames = [df1, df2]
df = pd.concat(frames, ignore_index=True)
# df = df1.append(df2)
print(len(df))

# Export
df.to_csv('training_data_sac_1.csv', index=False)