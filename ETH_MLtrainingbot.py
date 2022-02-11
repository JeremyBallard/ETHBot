import pandas as pd
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import IPython
import IPython.display
#from tensorflow.data.experimental import CsvDataset
#this is old, kinda jank
#tf.keras.preprocessing.timeseries_from_data_array does the same job but better
def candleProcessing(test_size, res_size):
	total_size = test_size+res_size
	res_ratio = res_size/total_size
	test_ratio = test_size/total_size
	run_size = 44005800/total_size
	trainArr = np.ndarray(shape=(int(run_size*test_ratio)+1, 
						test_size, 6), dtype=np.float64)
	resTrainArr = np.ndarray(shape=(int(run_size*test_ratio)+1, 
						res_size, 6), dtype=np.float64)
	testArr = np.ndarray(shape=(int(run_size*res_ratio)+1, 
						test_size, 6), dtype=np.float64)
	resTestArr = np.ndarray(shape=(int(run_size*res_ratio)+1, 
						res_size, 6), dtype=np.float64)
	raw_data = pd.read_csv('M:\Coding\Coinbase\candleData.csv')
	raw_data = raw_data.iloc[::-1]
	npData = raw_data.to_numpy(na_value=0, dtype=np.float64)
	for i in range(int(run_size)):
		if(i%(1/res_ratio) != 0):
			trainArr[i-int(i*res_ratio)-1] = npData[total_size*i:
						(total_size*i)+test_size]
			resTrainArr[i-int(i*res_ratio)] = npData[(total_size*i)+test_size:
						(total_size*i)+total_size]
		else:
			testArr[int(i*res_ratio)] = npData[total_size*i:
						(total_size*i)+test_size]
			resTestArr[int(i*res_ratio)] = npData[(total_size*i)+test_size:
						(total_size*i)+total_size]
	np.savez_compressed('trainData.npz', trainArr=trainArr, resTrainArr=resTrainArr)
	print("Successfully saved trainData.npz")
	np.savez_compressed('testData.npz', testArr=testArr, resTestArr=resTestArr)
	print("Successfully saved testData.npz")

df = pd.read_csv('M:\Coding\Coinbase\candleData.csv')
#don't need to average time element, unbased to do so
df.pop('time')
column_indices = {name: i for i, name in enumerate(df.columns)}
#necessary because data goes from newest at index 0 to oldest
#but that doesn't work with tf. .. .timeseries_from_data_array
#or with training
df = df.iloc[::-1]

n=len(df)
#needs to get changed to be moving average
train_df = df[0:int(n*0.7)]
#also val data should be compiled to be sections after each 
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean)/train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
df_std = (df - train_mean) / train_std
batch_size = 250
class WindowGenerator():
	def __init__(self, input_width, label_width, shift,
				train_df=train_df, val_df=val_df, test_df=test_df,
				label_columns=None, dtype=tf.float64):
		self.train_df = train_df
		self.val_df = val_df
		self.test_df = test_df

		self.dtype = dtype

		self.label_columns = label_columns
		if label_columns is not None:
			self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
		self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

		self.input_width = input_width
		self.label_width = label_width
		self.shift = shift

		self.total_window_size = input_width + shift

		self.input_slice = slice(0, input_width)
		self.input_indices = np.arange(self.total_window_size)[self.input_slice]

		self.label_start = self.total_window_size - self.label_width
		self.label_slice = slice(self.label_start, None)
		self.label_indices = np.arange(self.total_window_size)[self.label_slice]
	def __repr__(self):
		return '\n'.join([
			f'Total window size: {self.total_window_size}',
        	f'Input indices: {self.input_indices}',
        	f'Label indices: {self.label_indices}',
        	f'Label column name(s): {self.label_columns}'])

	def split_window(self, features):
		inputs = features[:, self.input_slice, :]
		labels = features[:, self.label_slice, :]
		if self.label_columns is not None:
			labels = tf.stack(
				[labels[:, :, self.column_indices[name]] for name in self.label_columns],
				axis=-1)
		inputs.set_shape([None, self.input_width, None])
		labels.set_shape([None, self.label_width, None])
		return inputs, labels

	def make_dataset(self, data):
		data = np.array(data, dtype=np.float64)
		ds = tf.keras.preprocessing.timeseries_dataset_from_array(
			data=data,
			targets=None,
			sequence_length=self.total_window_size,
			sequence_stride=1,
			shuffle=True,
			batch_size=batch_size)
		ds = ds.map(self.split_window)
		return ds

	@property
	def train(self):
		return self.make_dataset(self.train_df)
	@property
	def val(self):
		return self.make_dataset(self.val_df)
	@property
	def test(self):
		return self.make_dataset(self.test_df)

#this is because the data gets split into exactly batch_size amount of segments
#so going over this number is stupid and redundant
MAX_EPOCHS = batch_size
#patience tells how many epochs the model will run 
#without substantial improvement to the monitor val
def compile_and_fit(model, window, patience=5):

	callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
													patience=patience,
													mode='min'),
				tf.keras.callbacks.ModelCheckpoint(filepath='M:\Coding\ML\Model',
													save_weights_only=True,
													verbose=1,
													monitor='val_loss',
													mode='min',
													save_best_only=True)
				]
	model.compile(loss=tf.losses.MeanSquaredError(),
				optimizer=tf.optimizers.Adam(),
				metrics=[tf.metrics.MeanAbsoluteError()],
				run_eagerly=True)
	history = model.fit(window.train, epochs=MAX_EPOCHS,
						validation_data=window.val,
						callbacks=[callbacks])
	return history
#conv_width tells num of input vectors, out_steps is num of output vectors
CONV_WIDTH = 8
OUT_STEPS = 2
class FeedBack(tf.keras.Model):
	def __init__(self, units, out_steps):
		super().__init__()
		self.out_steps = out_steps
		self.units = units
		self.lstm_cell = tf.keras.layers.LSTMCell(units)
		self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
		#converting the intermediary back into a full tensor
		self.dense = tf.keras.layers.Dense(num_features)
	def warmup(self, inputs):
		x, *state = self.lstm_rnn(inputs)
		prediction = self.dense(x)
		return prediction, state
	#FeedBack.warmup = warmup
	#feeds output back as input and then allows another output to be predicted
	def call(self, inputs, training=None):
		predictions = []
		prediction, state = self.warmup(inputs)

		predictions.append(prediction)

		for n in range(1, self.out_steps):
			x = prediction
			x, state = self.lstm_cell(x, states=state,
										training=training)
			prediction = self.dense(x)
			predictions.append(prediction)
		predictions = tf.stack(predictions)
		#unsure if this is the right transposition for the given tensor
		predictions = tf.transpose(predictions, [1, 0, 2])
	#FeedBack.call = call


conv_window = WindowGenerator(input_width=CONV_WIDTH, label_width=OUT_STEPS, 
								shift=1, label_columns=['open', 'close'])
feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)
#print('Input shape:', conv_window.example[0].shape)
#print('Output shape:', multi_conv_model(conv_window.example[0]).shape)
history = compile_and_fit(feedback_model, conv_window)
IPython.display.clear_output()

#candleProcessing(15,5)
#with np.load('trainData.npz') as data:
	#trainData = data['trainArr']
	#resTrain = data['resTrainArr']
#train_ds = tf.data.Dataset.from_tensor_slices((trainData, resTrain))
#print("Loaded train_ds")
#with np.load('testData.npz') as data:
	#testData = data['testArr']
	#resTest = data['resTestArr']
#test_ds = tf.data.Dataset.from_tensor_slices((testData, resTest))
#print("Loaded test_ds")
#raw_data = pd.read_csv('M:\Coding\Coinbase\candleData.csv')
#print(raw_data.shape)
#data_package = TimeseriesGenerator(trainData, resTrain, length=1, 
				#batch_size=250
				#)
#model = 