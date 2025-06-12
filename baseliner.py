import pandas as pd


df = pd.read_csv("dataset/africanwildlife.csv")
from jedi import config, JediQA, get_dataloader
question = "how old are you?"

print(df.columns)
