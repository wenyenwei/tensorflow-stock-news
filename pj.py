
# ---------- Data Preprocessing ------------
import pandas as pd

df = pd.read_csv('all_stocks_5yr.csv') # load data
# only remain the columns we need
df = df.loc[(df['Name'] == "AAPL")] 
df = df.drop('volume', 1)
df = df.drop('high', 1)
df = df.drop('low', 1)
df = df.drop('Name', 1)
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna()

# rmls = [x for x in range(500,600)] + [y for y in range(1100,1200)] # train set
rmls = [x for x in range(500)] + [y for y in range(600,1100)] + [z for z in range(1200,1259)] # test set
df = df.drop(df.index[rmls])

# ---------- Training ------------
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import datetime
from tqdm import tqdm
import copy

class MainRNN():
    # ---------- Settings ------------
    def __init__(self):
        self.data_time_range=1
        self.seq_size=self.data_time_range
        self.hidden_layer=128
        self.output_feature_size=2
        self.epochs=1
        self.batch_size=64

    # ---------- Format Data ------------
    def x_y_to_seq(self, X):
        newX = []
        newY = []
        for i in range(int(len(X) - self.seq_size)):
            newX.append(X[i : (i + self.seq_size)])
            newY.append(X[(i + self.seq_size)])

        return newX, newY

    def get_formated_data(self):
        X = []
        Y = []
        for index, row in tqdm(df.iterrows()):
            row_data_array = row.values[1:]    
            X.append(row_data_array)
        
        X, Y = self.x_y_to_seq(X)
        
        return np.array(X), np.array(Y)


    # ---------- Define Vars ------------
    def process_train(self, X, Y):
        lr_to_use = [0.001 * (0.99 ** (batch/self.epochs)) for batch in range(self.epochs)]
        learning_rate = tf.placeholder(tf.float32, None)
        
        # get shape X (N, T, D)
        X_sample_size, X_seq_size, X_features_size = X.shape

        # get shape Y (K)
        Y_sample_size = Y.shape

        # init weight and bias
        weights = tf.Variable(tf.random_normal([self.hidden_layer, self.output_feature_size]))
        biases = tf.Variable(tf.random_normal([self.output_feature_size]))

        # placeholder for graph input
        tfX = tf.placeholder(tf.float32, shape=[None, X_seq_size, X_features_size], name='inputX')
        tfY = tf.placeholder(tf.float32, shape=[None, self.output_feature_size], name='inputY')

        # transposeX
        tfX = tf.transpose(tfX, [1, 0, 2])

        # define lstm cell
        lstmCell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_layer)
        lstmCell = tf.contrib.rnn.DropoutWrapper(lstmCell, output_keep_prob=0.8)

        # create RNN unit
        outputs, states = tf.nn.dynamic_rnn(cell=lstmCell, inputs=tfX, dtype=tf.float32)

        # get rnn output
        outputs = tf.stack(outputs)

        # transpose output back
        outputs = tf.transpose(outputs, [1, 0, 2])
        # outputs = tf.reshape(outputs, [outputs.get_shape()[-1], self.hidden_layer])

        # Hack to build the indexing and retrieve the right output.
        # Start indices for each sample
        index = tf.range(0, tf.shape(outputs)[0]) * X_seq_size + (X_seq_size - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_layer]), index)

        # prediction
        prediction = tf.matmul(outputs, weights) + biases
        label = tfY

        # cost function
        loss = tf.reduce_mean(tf.square(prediction - tfY))

        # optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        # cost[] and accuracies[]
        epoch_pred=[]
        epoch_lab=[]
        iterl=[]
        

        # global init
        init = tf.global_variables_initializer()

        first=True

        # ---------- Start Training ------------
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init)

            for epoch in range(self.epochs):
                costs=[]
                print('===== EPOCH ======: ', epoch)
                if not first:
                    saver.restore(sess,"./trained_model")
                    first=False
            
                c=0
                for batch in tqdm(range(X_sample_size - self.batch_size)):
                    curr_lr = lr_to_use[epoch]
                    batchX = X[batch:batch+self.batch_size]
                    batchY = Y[batch:batch+self.batch_size]

                    _, cost_out, prediction_out = sess.run([optimizer, loss, prediction], feed_dict={learning_rate: curr_lr, tfX: batchX.reshape(X_seq_size, self.batch_size, X_features_size), tfY: batchY.reshape(self.batch_size, self.output_feature_size)})
                    costs.append(cost_out)
                    if epoch == self.epochs-1:
                        epoch_pred.append(prediction_out[1])
                        epoch_lab.append(Y[batch+1])
                        iterl.append(c)
                        c+=1

                meanse = sum(costs) / float(len(costs))
                
                print('meanse: ', meanse)
                
                saver.save(sess, "./trained_model")
            plt.plot(iterl, epoch_pred, 'r--', iterl, epoch_lab, 'b--')
            plt.savefig('costs.png')

            np.random.seed(0)
            


    def run_prediction(self):
        print('start time:', datetime.datetime.now())
        X, Y = self.get_formated_data()
        self.process_train(X, Y)
        print('end time:', datetime.datetime.now())


if __name__ == '__main__':
    MainRNN().run_prediction()
