import requests

key = '7CV6MM8BME50NVP6'
# AAPL, IBM,
symbol = 'AAPL'
# monthly , 5min
interval = 'weekly'
#&time_period=14
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&outsize=full&interval={interval}&apikey={key}'
r = requests.get(url)
data = r.json()

print(data)
#NATR

#https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&outsize=full&interval=15min&apikey=7CV6MM8BME50NVP6