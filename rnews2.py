import tensorflow as tf
import pandas as pd
import numpy as np

class MainRNN():
	def __init__(self):
		self.data_time_range=7
		self.batch_size=self.data_time_range
		self.hidden_layer=1
		self.output_feature_size=1

	def x_y_to_seq(self, X, Y, batch_size):
		# X = [[[yesterday_stock_data(5)], [today_stock_data(5)], [tomorrow_stock_data(5)], ...batch_size], [repeat]]
		newX = []
		newY = []
		
		for i in range(len(X)):
			newX.append(X[i * self.batch_size : i * self.batch_size + self.batch_size])
			newY.append(Y[i * self.batch_size + self.batch_size - 1])

		return newX, newY

	def get_formated_data(self):
		
		df = pd.read_csv('all_stocks_5yr.csv')
		df = df.sort_values('date').reset_index(drop=True)
		
		# process different stock symbols
		df_symbols_encoded = pd.get_dummies(df, columns=['Name'], prefix=['symbol'])

		# match X and Y with date
		X = []
		Y = []
		for index, row in df_symbols_encoded.iterrows():
		    X.append(row.values[1:])
		    y_val = df.loc[(df['Name'] == "AAPL") & (df['date'] == row['date'])].values[0][4] # close price
		    Y.append(y_val)
		
		X, Y = x_y_to_seq(X, self.batch_size)

		# what should be the format of Y
		# regressor? classifier?

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
		tfX = tf.placeholder(tf.float32, shape=[None, X_seq_size, X_features_siz], name='inputX')
		tfY = tf.placeholder(tf.float32, shape=[None, self.output_feature_size], name='inputY')

		# transposeX
		tfX = tf.transpose(tfX, [1, 0, 2])

		# target = reshape Y

		# define lstm cell
		lstmCell = tf.contrib.nn.BasicLSTMCell(self.hidden_layer)

		# create RNN unit
		outputs, states = tf.contrib.nn.rnn(cell=lstmCell, inputs=tfX, dtype=float32, sequence_length=self.batch_size)

		# get rnn output
		outputs = tf.stack(outputs)

		# transpose output back
		outputs = tf.transpose(outputs, [1, 0, 2])
		tf.reshape(outputs, [-1, self.hidden_layer])

		# model(logits)


		# prediction
		prediction = tf.matmul(outputs, weights) + biases

		# cost function
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=tfY))

		# optimizer
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

		# evaluate model
		correct_pred = tf.math.less(tf.math.abs(pred - tfY), tf.math.multiply(tfY))
		accuracy = tf.math.divide(correct_pred, Y_sample_size)

		# global init
		init = tf.global_variables_initializer()

		# start training
		with tf.Session() as sess:

			sess.run(init)





	def run_prediction():
		X, Y = get_formated_data()
		process_train(X, Y)


if __name__ == '__main__':
	run_prediction()
