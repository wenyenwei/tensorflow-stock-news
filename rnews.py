import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


tf.reset_default_graph()
lstm_graph = tf.Graph()

# Dataset downloaded from kaggle: https://www.kaggle.com/camnugent/sandp500/home
df = pd.read_csv('all_stocks_5yr.csv') 
df = df.sort_values('date')


# Split data into test, train set
X = []
Y = []

df_grouped = df.groupby('date', as_index=False).sum()
# start from here again
for index, row in df_grouped.iterrows():
    X.append(row.values[1:])
    # change to 5 again
    y_val = df.loc[(df['Name'] == "AAPL") & (df['date'] == row['date'])].values[0][1:6]
    Y.append(y_val)
    print (index)
      
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20)

###################
hidden_size = 128
num_of_layers = 3
batch_size = 1
max_epoch = 50
num_of_step = 5
input_size = 1

with lstm_graph.as_default():    
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(hidden_size)
    
    stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell() for _ in range(num_of_layers)
            ]) if num_of_layers > 1 else lstm_cell()
    
    words = tf.placeholder(tf.float32, [None, num_of_step, input_size])
    target = tf.placeholder(tf.float32, [None, input_size])

    weight = tf.Variable(tf.truncated_normal([num_of_step, input_size]))
    bias = tf.Variable(tf.constant(0.1, shape=[input_size]))
    
    
    data, _ = tf.nn.dynamic_rnn(cell=stacked_lstm_cell, inputs=words, dtype=tf.float32)
    data = tf.transpose(data, [1, 0, 2])
    last = tf.gather(data, int(data.get_shape()[0]) - 1, name="last_lstm_output")
    
    prediction = tf.matmul(weight, last) + bias
    
    loss = tf.norm(prediction - target, ord='euclidean') #, axis=None, keepdims=None, name=None, keepdims=None
    # loss = tf.reduce_mean(tf.square(prediction - targets))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

with tf.Session(graph=lstm_graph) as sess:
    tf.global_variables_initializer().run()

    for i in range(len(Xtrain)):
        current_loss, _ = sess.run(
            [loss, optimizer], 
            feed_dict={
                words: [Xtrain[i].reshape(5,1)],
                target: Ytrain[i].reshape(5,1)
            }
        )
        print("current_loss", current_loss)

    saver = tf.train.Saver()
    saver.save(sess, "./trained_model", global_step=max_epoch)

    ### Predictions
    saver.restore(sess,"./trained_model-50")
    
    for i in range(len(Xtest)):
        
        y_pred = sess.run(
                data,
                feed_dict={
                        words: [Xtest[i].reshape(5,1)]
                        }
                )
        
        print("y_pred", y_pred)
        
