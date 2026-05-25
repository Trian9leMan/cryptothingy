import requests
import pandas as pd

def updatecryptodata():
    api_key = 'W1QRDEILW9P5P95Y'
    url = (f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey={api_key}')

    response = requests.get(url)
    datatd = response.json()

    df = pd.DataFrame.from_dict(datatd['Time Series (Digital Currency Daily)'],orient='index')

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df['close'] = df['4. close'].astype(float)
    df['open'] = df['1. open'].astype(float)
    df['high'] = df['2. high'].astype(float)
    df['low'] = df['3. low'].astype(float)
    df['volume'] = df['5. volume'].astype(float)

    df = df.reset_index().rename(columns={'index': 'date'})
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

    try:
        old = pd.read_csv('cryptodata.csv')
    except FileNotFoundError:
        df.to_csv('cryptodata.csv', index=False)
        old = df

    old['date'] = pd.to_datetime(old['date'])
    df['date'] = pd.to_datetime(df['date'])

    if df['date'].max() > old['date'].max():
        combined = pd.concat([old, df])
        combined = combined.drop_duplicates(subset='date')
        combined = combined.sort_values('date')
        combined.to_csv('cryptodata.csv', index=False)

        return combined
    else:
        return old