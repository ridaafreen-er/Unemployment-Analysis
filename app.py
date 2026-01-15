import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- APP TITLE ----------------
st.title("📊 Unemployment Data Science Dashboard")
st.write("Analysis & Prediction of Unemployment Rate in India")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Unemployment in India.csv")
df.columns = df.columns.str.strip()

data = df[['Date', 'Region', 'Estimated Unemployment Rate (%)']].copy()

# Convert 'Date' with dayfirst=True to avoid parsing issues
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Estimated Unemployment Rate (%)'] = pd.to_numeric(
    data['Estimated Unemployment Rate (%)'], errors='coerce'
)
data.dropna(inplace=True)
data['Region'] = data['Region'].str.strip()

# ---------------- SIDEBAR FILTER ----------------
st.sidebar.header("Filters")
selected_region = st.sidebar.selectbox(
    "Select Region",
    data['Region'].unique()
)

region_data = data.loc[data['Region'] == selected_region].copy()

# ---------------- DATA VISUALIZATION ----------------
st.subheader(f"Unemployment Trend - {selected_region}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    region_data['Date'],
    region_data['Estimated Unemployment Rate (%)'],
    marker='o'
)
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.grid()
st.pyplot(fig)

# ---------------- FEATURE ENGINEERING ----------------
region_data['Year'] = region_data['Date'].dt.year
region_data['Month'] = region_data['Date'].dt.month

X = region_data[['Year', 'Month']]
y = region_data['Estimated Unemployment Rate (%)']

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = model.score(X_test, y_test)

st.subheader("📈 Model Performance")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# ---------------- FUTURE PREDICTION ----------------
st.subheader("🔮 Predict Future Unemployment")

year = st.number_input(
    "Year",
    min_value=int(region_data['Year'].max()),
    max_value=int(region_data['Year'].max()) + 1,
    value=int(region_data['Year'].max())
)
month = st.slider("Month", 1, 12, 1)

future = pd.DataFrame({'Year': [year], 'Month': [month]})
prediction = model.predict(future)

# Clip prediction between 0-100% for realism
prediction_value = max(0, min(prediction[0], 100))

st.success(
    f"Predicted Unemployment Rate for {selected_region}: {prediction_value:.2f}%"
)
st.info(
    "Note: Linear regression is a baseline model. Predictions far beyond historical data may be inaccurate."
)

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Data Preview")
st.dataframe(region_data)
