import pandas as pd

def summarize(df):
    print("\n📌 Data Summary")
    print(df.head())
    print(df.info())
    print(df.describe())
