import pandas as pd

# MRJ: I have moved to this file the methods and code
# commented out in rnews2.py

class Dataset(object):

    def __init__(self):
		# 88434
		self.current_index=6
		self.current_escape_index=self.current_index - 1
		self.data_point=50*self.current_index
		self.data_size=self.seq_size*self.data_point

    def prepare(self):
        df = pd.read_csv('all_stocks_5yr.csv')
        df = df.sort_values('date').reset_index(drop=True)

        # process different stock symbols
        df_symbols_encoded = pd.get_dummies(df, columns=['Name'], prefix=['symbol'])[:self.data_size]

        # match X and Y with date
        X = []
        Y = []
        for index, row in df_symbols_encoded.iterrows():
        	if index >= 50*7*(self.current_escape_index):
        	    X.append(row.values[1:])
        	    y_val = df.loc[(df['Name'] == "AAPL") & (df['date'] == row['date'])].values[0][4] # close price
        	    Y.append(y_val)
        	    print(index)

        self.save_to_csv(X, Y)

	def save_to_csv(self, X, Y):
		with open('X_preprocessed_data.csv', 'ab') as f:
			for row in range(len(X)):
				np.savetxt(f, X, delimiter=",")


		with open('Y_preprocessed_data.csv', 'ab') as f:
			for row in range(len(Y)):
				np.savetxt(f, Y, delimiter=",")

def main():

    dataset = Dataset()
    dataset.prepare()

if __name__ == '__main__':
    main()
