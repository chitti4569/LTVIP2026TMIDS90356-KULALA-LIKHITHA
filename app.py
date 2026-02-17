import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import requests

path = "data/Location1.csv"
df = pd.read_csv(path, encoding="latin1")

df.columns = df.columns.str.strip()

df.rename(columns={
    'Date/Time': 'Time',
    'LV ActivePower (kW)': 'ActivePower_kW',
    'Wind Speed (m/s)': 'WindSpeed',
    'Wind Direction (°)': 'WindDirection'
}, inplace=True)

sns.pairplot(df)
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
ax = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
bottom, top = ax.get_ylim()
plt.show();
ax.set_ylim(bottom + 0.5, top - 0.5)
print(corr)

df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

df["Hour"] = df["Time"].dt.hour
df["Month"] = df["Time"].dt.month
df["DayOfYear"] = df["Time"].dt.dayofyear

required_cols = ["ActivePower_kW", "WindSpeed"]

if "Theoretical_Power_Curve (KWh)" in df.columns:
    required_cols.append("Theoretical_Power_Curve (KWh)")
    print("✔ Using Theoretical Power Curve feature")
else:
    print("⚠ Theoretical Power Curve column NOT found")

df = df.dropna(subset=required_cols)

features = ["WindSpeed", "Hour", "Month", "DayOfYear"]

if "Theoretical_Power_Curve (KWh)" in df.columns:
    features.insert(1, "Theoretical_Power_Curve (KWh)")

X = df[features]
y = df["ActivePower_kW"]

train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=0.25, shuffle=False
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(train_X, train_y)

preds = model.predict(val_X)

print("\nModel Performance")
print("MAE:", mean_absolute_error(val_y, preds))
print("R² :", r2_score(val_y, preds))


joblib.dump(model, "power_prediction_model.pkl")
print("\nModel saved as power_prediction_model.pkl")

app = Flask(__name__)
model = joblib.load('power_prediction_model.pkl')


@app.route('/')
def home():
    return render_template('intro.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/windapi', methods=['POST'])
def windapi():
    city = request.form.get('city')
    apikey = 'f54119f50d7337ac8de52db5cc2fbd91'
    url = 'http://api.openweathermap.org/data/2.5/weather?q=' + city + "&appid=" + apikey
    resp = requests.get(url)
    resp = resp.json()

    temp = str(resp["main"]["temp"]) + "°C"
    humid = str(resp["main"]["humidity"]) + "%"
    pressure = str(resp["main"]["pressure"]) + " mmHg"
    speed = str(resp["wind"]["speed"]) + " m/s"

    return render_template('predict.html', temp=temp, humid=humid, pressure=pressure, speed=speed)


@app.route('/y_predict', methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[float(x) for x in request.form.values()]]

    prediction = model.predict(x_test)
    print(prediction)
    output = prediction[0]

    return render_template('predict.html', prediction_text='The energy predicted is {:.2f} KWh'.format(output))
if __name__ == "__main__":
    app.run(debug=False)