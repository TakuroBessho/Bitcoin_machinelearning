# ライブラリの読み込み
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import time
import requests
import json
from datetime import datetime 
import talib
import sklearn 
from sklearn.model_selection import train_test_split 
from sklearn import tree

#=====前処理

def get_btcprice(ticker, max):
    url = ('https://api.coingecko.com/api/v3/coins/')+ticker+('/market_chart?vs_currency=jpy&days=')+max
    r = requests.get(url)
    r2 = json.loads(r.text)
    return r2 
    
def get_price(r2): 
    s = pd.DataFrame(r2['prices'])
    s.columns = ['date', 'price']
    date = [] 
    for i in s['date']: 
        tsdate = int(i / 1000) 
        loc = datetime.utcfromtimestamp(tsdate) 
        date.append(loc) 
        s.index = date 
        del s['date'] 
        return s 

# ビットコインの全期間の価格データを取得する 
r2 = get_btcprice('bitcoin', 'max') 
btc = get_price(r2)

# 対数収益率の計算 
change = btc['price'].pct_chnage()

# talibでテクニカル指標を計算する
price = btc['price']
momentam = round(talib.MOM(price, 5), 0)
macd = talib.MACD(price)
rsi = round(talib.RSI(price, timeperiod=7), 0)

# 各データをつなぎ合わせてデータフレームを作成
df = pd.DataFrame({"date": btc.index, "price": change, "mom": momentam, "macd": round(macd[2], 0), "rsi": rsi})

# 説明変数xと被説明変数ｙを決める
y = df['price'][35:]
x = df[['rsi','mom','macd']][34:-1]

# xとyの要素数が同じかを確認
print(len(x), len(y))

# 変化率を1と0のシグナルに変換する
signal = []
for i in y: 
    if i > 0: 
        signal.append(1) 
    elif i < 0: 
        signal.append(-1)

# macd momentumが0以上ならば1, 0以下ならば0に置き換える 
x.loc[df['macd'] < 0, 'macd'] = -1 
x.loc[df['macd'] >= 0, 'macd'] = 1 
x.loc[df['mom'] < 0, 'mom'] = -1 
x.loc[df['mom'] >= 0, 'mom'] = 1 
x.loc[df['rsi'] < 50, 'rsi'] = -1 
x.loc[df['rsi'] >= 50, 'rsi'] = 1 

# 機械学習用に次元を変換する
y2 = np.array(signal).reshape(-1,) 

# xとyの要素数が同じか再確認 
print(len(x), len(y2))

# データを9:1に分割する 
(X_train, X_test,y_train, y_test) = train_test_split(x, y2, test_size=0.1, random_state=0, shuffle=False)


#==========機械学習============#

ライブラリの読み込み&クラス呼び出し&インスタンスの生成 
clf = tree.DecisionTreeClassifier(max_depth=5)

# 学習開始！
clf = clf.fit(X_train.values, y_train)

# 作成した機械学習モデルをテストデータに当てはめる
predicted = clf.predict(X_test)

# モデルのテストデータに対する精度を確認
score = sum(predicted == y_test) / len(y_test)
print('モデルの精度は{}%です'.format(score * 100))



#======機械学習の結果に基づいてトレードした場合の収益を計算する========#

# 予測データの長さを計算する
length = len(predicted)

# 機械学習の予測結果に従ってトレードした場合のリターンを計算する
ai_return = (y[-length:] * predicted + 1).cumprod()

# 同じ期間ホールドしていた場合のリターンを計算する
hold_return = (y[-length:] + 1).cumprod() 

# 各累積収益率をプロットする
ai_return.plot(label='ai_trade', legend=True)
hold_return.plot(label='hold', legend=True)
