import logging # MRJ: for creating logs
import argparse # MRJ: for parsing program arguments
from tqdm import tqdm # MRJ: for progress bars

import tensorflow as tf
import pandas as pd
import numpy as np

# MRJ: I advise to use Jupyter notebooks for visualization and analysis
# of results
#import matplotlib.pyplot as plt

from sklearn.utils import shuffle


# TODO:
# 1. rewrite pre-processing
# 2. fix batch problem

class MainRNN():

    # MRJ: note the use of kwargs
    def __init__(self, **kwargs):
            #self.data_time_range=7
    # MRJ: using 7-grams for training the RNN implies that we
    # have an astronomical number of possible input sequences...
    # Let's try first with 2, which is equivalent to assume
    # that the time series is a Markov chain. This assumption
    # may be wrong, as we do not capture every factor that affects
    # stock prices with the features used in the training set.
    # Remains to be seen how wrong it is.
    self.data_time_range=2
        self.seq_size=self.data_time_range
        self.hidden_layer=1
        self.output_feature_size=1
        self.epochs= kwargs['epochs']

        self.error_rate=0.1
        self.batch_size=64

    def x_y_to_seq(self, X, Y):
        # X = [[[yesterday_stock_data(5)], [today_stock_data(5)], [tomorrow_stock_data(5)], ...batch_size], [repeat]]
        newX = []
        newY = []
        for i in tqdm(range(int(len(X) / self.seq_size))):
            #print((i + 1) * self.seq_size - 1)
            newX.append(X[i * self.seq_size : (i + 1) * self.seq_size])
            newY.append(Y[(i + 1) * self.seq_size - 1])

        return newX, newY

    def get_formated_data(self):

        dfX = pd.read_csv('X_preprocessed_data.csv.gz', header=None)
        dfY = pd.read_csv('Y_preprocessed_data.csv.gz', header=None)

        logging.info("Arranging training data into sequences...")
        X, Y = self.x_y_to_seq(dfX.values, dfY.values)

        # what should be the format of Y
        # regressor? classifier?

        return np.array(X), np.array(Y)


    def process_train(self, X, Y):

        # get shape X (N, T, D)
        X_sample_size, X_seq_size, X_features_size = X.shape
    logging.info("Size of training data set: {}".format(X_sample_size))
    logging.info("Size of sequence: {}".format(X_seq_size))
    logging.info("Feature set size: {}".format(X_features_size))
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
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=tfY))
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

            for epoch in tqdm(range(self.epochs)):
                logging.info("Epoch {} started".format(epoch))

                X, Y = shuffle(X, Y)
                cost = 0
                accuracy = 0
                for batch in range(X_sample_size - self.batch_size):
                    batchX = X[batch:batch+self.batch_size]
                    batchY = Y[batch:batch+self.batch_size]

                    _, cost_out, prediction_out = sess.run([optimizer, loss, prediction], feed_dict={tfX: batchX.reshape(X_seq_size, self.batch_size, X_features_size), tfY: batchY.reshape(self.batch_size, self.output_feature_size)})

                    correct_pred = tf.cast(tf.less(tf.cast(tf.abs(prediction_out - batchY), tf.float64), tf.multiply(batchY, self.error_rate)), tf.float32)
                    cost += cost_out
                    accuracy += correct_pred.eval()[0][0]
                    # print('prediction_out', prediction_out)
                    # print('batchY', batchY)
                    # print('correct_pred.eval()[0][0]', correct_pred.eval()[0][0])

                costs.append(cost)
                accuracies.append(accuracy / (batch + 1))
                # MRJ: We log the last value of each of these
                logging.info('Epoch {} cost: {}'.format(epoch, costs[-1]))
                logging.info('Epoch {} accuracy: {}'.format(epoch, accuracies[-1]))
                # print('cost: ', cost)
                # print('accuracy: ', accuracy)
                # print('batch: ', batch)
                # print('accuracy / batch', accuracy / (batch + 1))

        # MRJ: see notebook 'notebooks/tutorial/000 - Introduction to Jupyter Notebooks.ipynb'
        #plt.plot(costs)
        #plt.show()
        logging.info("Storing training performance")
        # MRJ: Next we create a pandas dataframe from a dictionary
        performance_df = { 'epoch' : [k for k in range(self.epochs)],
                                                'cost' : costs,
                                                'accuracy' : accuracies}
        performance_df = pd.DataFrame(performance_df)
        performance_df.to_csv("training_performance.csv")

    def run_prediction(self):
        logging.info("Loading training and test data...")
        print("Loading training and test data...")
        X, Y = self.get_formated_data()
        logging.info("Processing training data...")
        print("Proccessing training data...")
        self.process_train(X, Y)


def parse_arguments():
    # MRJ: the argparse module is the Python standard library module to
    # process command line arguments
    # https://docs.python.org/3/howto/argparse.html

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", help="Sets the seed of the random number generator (defaults: 1337)", type=int, default=1337)
    parser.add_argument("--debug", help="Activate verbose logging")
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=3)

    return parser.parse_args()

def main():
    args = parse_arguments()

    # MRJ: we setup the behaviour of the logging module
    log_level = logging.INFO
    if args.debug:
        # MRJ: when the debug flag is on we activate more verbose logging
        log_level = logging.DEBUG

    # @TODO: make the name of the log file to be configurable via the
    # command line
    logging.basicConfig(filename='rnews2.log', format='%(levelname)s:%(asctime)s: %(message)s', level=log_level)


    # MRJ: One important detail is to always fix the seed of the training
    # process for repeatibility. Learning algorithms are most often applications
    # of stochastic optimization, so you can definitely get substantially
    # different results across several runs. Fixing the seed of the RNG
    # allows you to get explicit control over this, replicate experiments,
    # and most importantly, perform simple meta-optimization by running
    # several times the training process and then selecting the "best"
    # performing set of parameters.
    np.random.seed(args.seed)
    logging.debug("RNG seed fixed to: {}".format(args.seed))

    logging.info("Training started...")
    print("Training started...")

    # MRJ: note the use of kwargs
    MainRNN(epochs=args.epochs).run_prediction()

if __name__ == '__main__':
    # MRJ: note that I am using an old (and in my opinion useful) idiom to
    # structure Python scripts
    main()
