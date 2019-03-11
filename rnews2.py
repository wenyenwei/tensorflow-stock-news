import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer


class MainRNN():
	def __init__(self):
		self.data_time_range=7
		self.batch_size=self.data_time_range
		self.hidden_layer=1
		self.output_feature_size=1
		self.epochs=3
		self.data_size=self.batch_size*100
		self.error_rate=0.1

	def x_y_to_seq(self, X, Y):
		# X = [[[yesterday_stock_data(5)], [today_stock_data(5)], [tomorrow_stock_data(5)], ...batch_size], [repeat]]
		newX = []
		newY = []
		for i in range(int(len(X) / self.batch_size)):
			print((i + 1) * self.batch_size - 1)
			newX.append(X[i * self.batch_size : (i + 1) * self.batch_size])
			newY.append(Y[(i + 1) * self.batch_size - 1])

		return newX, newY

	def get_formated_data(self):
		
		df = pd.read_csv('all_stocks_5yr.csv')
		df = df.sort_values('date').reset_index(drop=True)
		df_text = pd.read_csv('rnews_apple.txt', sep=': \"', header=None, names=['date', 'text'], dtype='str', engine='python')
		count_vectorizer = CountVectorizer(decode_error='ignore',binary='boolean')
		X_text = count_vectorizer.fit_transform(df_text['text'])
		X_text = X_text.toarray()

		# process different stock symbols
		df_symbols_encoded = pd.get_dummies(df, columns=['Name'], prefix=['symbol'])[:self.data_size]

		# match X and Y with date
		X = []
		Y = []
		for index, row in df_symbols_encoded.iterrows():
			row_data_array = row.values[1:]
			row_day = row['date'].split('-')[2]
			row_mon = row['date'].split('-')[1]
			row_year = row['date'].split('-')[0]
			text_index = df_text[(df_text['date'] == row_mon + row_day + row_year)].index.values.astype(int)[0]
			if (text_index) >= 0: 
				row_data_array = np.append(row_data_array, np.array(X_text[text_index], dtype=np.int))
			else:
				_, shape_for_use = np.array(X_text).shape
				row_data_array = np.append(row_data_array, np.zeros(shape_for_use, dtype=np.int))
			print("row_data_array.shape", row_data_array.shape)
			X.append(row_data_array)
			y_val = df.loc[(df['Name'] == "AAPL") & (df['date'] == row['date'])].values[0][4] # close price
			Y.append(y_val)
			print(index)
		
		X, Y = self.x_y_to_seq(X, Y)

		return np.array(X), np.array(Y)

		
	def process_train(self, X, Y):

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

		# target = reshape Y

		# define lstm cell
		lstmCell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_layer)

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

		# model(logits)


		# prediction
		prediction = tf.matmul(outputs, weights) + biases

		# cost function
		loss = tf.reduce_mean([prediction ,tfY])

		# optimizer
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

		# evaluate model
		correct_pred = tf.cast(tf.less(tf.abs(prediction - tfY), tf.multiply(tfY, self.error_rate)), tf.float32)
		# accuracy = tf.divide(correct_pred, Y_sample_size)

		# cost[] and accuracies[]
		costs = []
		accuracies = []

		# global init
		init = tf.global_variables_initializer()

		# start training
		with tf.Session() as sess:

			sess.run(init)

			for epoch in range(self.epochs):

				X, Y = shuffle(X, Y)				
				accuracy = 0
				for batch in range(X_sample_size):
					batchX = X[batch]
					batchY = Y[batch]
                    
					_, cost_out, correct_pred_out = sess.run([optimizer, loss, prediction], feed_dict={tfX: batchX.reshape(X_seq_size, 1, X_features_size), tfY: batchY.reshape(1, self.output_feature_size)})
					
					correct_pred = tf.cast(tf.less(tf.cast(tf.abs(correct_pred_out - batchY), tf.float64), tf.multiply(batchY, self.error_rate)), tf.float32)
					accuracy += correct_pred.eval()[0][0]
					print('cost_out', cost_out)

					costs.append(cost_out)
				accuracies.append(accuracy / (batch + 1))

		plt.plot(costs)
		plt.show()

	def run_prediction(self):
		X, Y = self.get_formated_data()
		print("X.shape", X.shape)
		self.process_train(X, Y)


if __name__ == '__main__':
	X, Y = MainRNN().run_prediction()
