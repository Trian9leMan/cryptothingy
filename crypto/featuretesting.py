import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import csvfetch
import numpy as np
from itertools import combinations

def buysell(df):
    df['return_1'] = df['close'].pct_change()
    df['return_3'] = df['close'].pct_change(3)
    df['return_7'] = df['close'].pct_change(7)

    df['ma7'] = df['close'].rolling(7).mean()
    df['ma14'] = df['close'].rolling(14).mean()
    df['ma30'] = df['close'].rolling(30).mean()

    df['price_ma7_gap'] = df['close'] - df['ma7']
    df['price_ma30_gap'] = df['close'] - df['ma30']

    df['volatility7'] = df['return_1'].rolling(7).std()
    df['volatility14'] = df['return_1'].rolling(14).std()

    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['volume_change'] = df['volume'].pct_change()

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    features = ['open', 'high', 'low', 'close', 'volume', 'return_1', 'return_3', 'return_7', 'ma7', 'ma14', 'ma30', 'price_ma7_gap', 'price_ma30_gap', 'volatility7', 'volatility14', 'hl_range', 'volume_change']
    featurescombs = ()
    scoresandcombs = {}
    for i in range(1, len(features)+1):
        featurescombs.extend(combinations(features, i))

        for comb in featurescombs:
            comb = len(featurescombs)
        

    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    min_child_weight=3,
    gamma=0.1,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1)
    model.fit(X_train, y_train)

    trainingacc = model.score(X_train, y_train)
    print("Training Accuracy:", trainingacc)

    testingacc = model.score(X_test, y_test)
    print("Testing Accuracy:", testingacc)

    df.loc[X_test.index, 'prediction'] = model.predict(X_test)
    return df

csvfetch.updatecryptodata()
df = pd.read_csv('cryptodata.csv')
buysell(df)