import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_data(ticker='AAPL'):
    df = yf.download(ticker, period='6mo', interval='1d', group_by='column')
    df.reset_index(inplace=True)
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'] > 0).astype(int)
    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.2f}")
    return model

if __name__ == "__main__":
    df = prepare_data()
    model = train_model(df)
