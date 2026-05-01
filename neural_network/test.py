import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# df = pd.read_csv("train.csv")

# print(df.head())

# # Example: show image from row 0


# import pandas as pd

# labels_df = pd.read_csv("hasy-data-labels.csv")
# print(labels_df.columns)
# print(labels_df.head())
import pandas as pd

# Load the CSV
labels_df = pd.read_csv("hasy-data-labels.csv")

# Columns you want to inspect
columns_to_show = ["symbol_id", "latex", "path"]

# Filter for the math symbols you care about
target_symbols = ["+", "-", "\\times", "\\div", "="]
filtered_df = labels_df[labels_df["latex"].isin(target_symbols)]

# Print the filtered table
print(filtered_df[columns_to_show])
