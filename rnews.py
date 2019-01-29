import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing


# Dataset downloaded from kaggle: https://www.kaggle.com/camnugent/sandp500/home
df = pd.read_csv('all_stocks_5yr.csv') 
df = df.sort_values('date')


# draw image of the data
df = df.groupby('date').sum()

plt.figure(figsize=(18,9))
plt.plot(range(df.shape[0]), ((df['low']+df['high'])/2.0/500)*df['volume'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Market Capital',fontsize=18)
plt.show()

# transform texts to vector and store them into a map
df_news = pd.read_csv('rnews_apple.txt', sep=": \"", header=None)
df_news.columns = ['date', 'news']
count_vectorizer = CountVectorizer(decode_error='ignore',binary='boolean')
df_news['news_vector'] = count_vectorizer.fit_transform(df_news['news'])


news_map = {}
for index, row in df_news.iterrows():
    original_date = str(row['date'])
    if len(original_date) == 7:
        original_date = "0" + original_date

    transformed_date = original_date[4:8] + "-" + original_date[0:2] + "-" + original_date[2:4]
    news_map[transformed_date] = row['news_vector']

# Split data into test, train set
X = []
Y = []

df_grouped = df.groupby('date', as_index=False).sum()
for index, row in df_grouped.iterrows():
    row['news'] = news_map[row['date']] if 'date' in news_map else ""
    X.append(row.values)
    Y.append(df.loc[(df['Name'] == "AAPL") & (df['date'] == row['date'])].values)
    print (index, row['date'])
    
# for index, row in df.iterrows():
  #  if row['Name'] != "AAPL":
   #     row['news'] = news_map[row['date']] if 'date' in news_map else ""
    #    X.append(row.values)
     #   Y.append(df.loc[(df['Name'] == "AAPL") & (df['date'] == row['date'])].values)
      #  print(index, row['date'])
      
      
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20)

# Configuration is wrapped in one object for easy tracking and passing.
class RNNConfig():
    input_size=1
    num_steps=30
    lstm_size=128
    num_layers=1
    keep_prob=0.8
    batch_size = 64
    init_learning_rate = 0.001
    learning_rate_decay = 0.99
    init_epoch = 5
    max_epoch = 50

config = RNNConfig()

import tensorflow as tf
tf.reset_default_graph()
lstm_graph = tf.Graph()

with lstm_graph.as_default():
# Dimension = (
    #     number of data examples, 
    #     number of input in one computation step, 
    #     number of numbers in one input
    # )
    # We don't know the number of examples beforehand, so it is None.
    inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size])
    targets = tf.placeholder(tf.float32, [None, config.input_size])
    learning_rate = tf.placeholder(tf.float32, None)

def _create_one_cell():
    return tf.contrib.rnn.LSTMCell(config.lstm_size, state_is_tuple=True)
    if config.keep_prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)


























