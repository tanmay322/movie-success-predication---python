import pandas as pd
import numpy as np
import re
import csv
imdb_data = pd.read_csv("movie_metadata.csv")
cleaned_data = imdb_data.set_index('movie_title').genres.str.split('|', expand=True).stack()
cleaning = pd.get_dummies(cleaned_data).groupby(level=0, sort=False).sum()
print cleaning

modified = pd.DataFrame (data=cleaning)
modified.to_csv("output_binary_genres.csv")
