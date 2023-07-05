import requests, time, csv
import datetime as dt
import itertools
from coinbase.wallet.client import Client

URL = 'https://api.exchange.coinbase.com'
sesh = requests.Session()

#at some point convert to environmental variables
#Fill these out
ID = 
passphrase = 
SECRET = 

#necessary to fetch time from API server, no chance of mismatch, sets up payload for future use
client = Client(ID, SECRET)
payload = {}

def candleQuery(payload):
	#change method and requestPath as necessary for each command to API
	requestPath = '/products/ETH-USD/candles'
	#Accept:app/json is necessary for every request handled by requests for this, everything else is standard to coinbase protocol
	headers = {"Accept": "application/json"}
	response = sesh.get(URL + requestPath, params=payload, headers=headers)
	#prints status code, 200 is goal
	#print(response.status_code)
	return response.json()
#Used for initilization of candle grabbing, have to put 299 for max candle grab
def candlePayloadInit():
	#gets the time and moves it to most recent valid 60 second interval
	timestamp = client.get_time()
	time = timestamp.epoch - (timestamp.epoch % 60)
	#startOfLast is used for next loop in int, then translates into iso format
	startOfLast = time - (299*60)
	start = dt.datetime.utcfromtimestamp(startOfLast).isoformat()
	end = timestamp.iso
	payload = {"granularity": "60", "start": start, "end": end, "startOfLast": startOfLast}
	return payload
#payload to use for after initialization
def candlePayload(startOfLast):
	startTime = startOfLast - (299*60)
	start = dt.datetime.utcfromtimestamp(startTime).isoformat()
	end = dt.datetime.utcfromtimestamp(startOfLast).isoformat()
	payload = {"granularity": "60", "start": start, "end": end,"startOfLast": startOfLast}
	return payload

header = ['time', 'low', 'high', 'open', 'close', 'volume']

#initialization of candles
OLpayload = candlePayloadInit()
#slices first 3 parts of payload for URL
payload = dict(itertools.islice(OLpayload.items(), 3))
candleAggr = candleQuery(payload)
f = open('candleData.csv', 'w')
writer = csv.writer(f)
writer.writerow(header)
f.close()
with open('candleData.csv', 'a', newline='') as f:
	writer = csv.writer(f)
	writer.writerows(candleAggr)
#last part of payload used for next loop
startOfLast = OLpayload["startOfLast"]
#loopable portion of candles
for i in range(155520):
	OLpayload = candlePayload(startOfLast)
	payload = dict(itertools.islice(OLpayload.items(), 3))
	candleAggr = candleQuery(payload)
	startOfLast = OLpayload["startOfLast"]
	with open('candleData.csv', 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerows(candleAggr)
#candle data testing
#print(candleAggr[0])
#print(candleAggr[0][4])
#candleUpOrDn = candleAggr[0][4] - candleAggr[0][3]
#print(candleUpOrDn)
