ETHBot takes candle data and tries to build a model that predicts future movement. Current ideal is 15 minutes of data for 5 minutes of prediction, but training a model on my system is not feasible for now.
Has two differing ways to preprocess data. candleProcessing is an older, hacked together way before I discovered the time series preprocessing in TensorFlow.
Both rely on a dataset that is not present in the git, because it's 2.5GB (though a zip exists that's 18MB). The dataset is 1-minute candle data downloaded from Coinbase API going from about July 2020 to late December 2021.
Should be uploading dataset soon to a server so the code can download locally and preprocess the data like that. Uploading to a server would help for continuous training since the server can collect new candle data and then send a daily chunk for the model to train on. 
Built using pandas, Tensorflow with Keras, numpy, IPython and time libraries.
