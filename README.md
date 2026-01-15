# 📊 Unemployment Rate Prediction Dashboard - India

This project is an **interactive Data Science dashboard** that analyzes and predicts the **unemployment rate in India**. It is built using **Python, Streamlit, and Linear Regression**, allowing users to explore historical trends and forecast future unemployment rates for different regions.

---

## 🔹 Features

- **Data Analysis & Visualization**
  - Cleaned and preprocessed unemployment data
  - Visualized historical unemployment trends for each region
- **Machine Learning Prediction**
  - Built a **Linear Regression** model to predict future unemployment rates
  - Evaluated model performance using **MAE, RMSE, and R² Score**
- **Interactive Dashboard**
  - Select regions from the sidebar
  - View historical trend plots
  - Input year and month to predict future unemployment
- **Data Preview**
  - View top rows of cleaned dataset

---

## 🔹 Tech Stack

- **Python** – Data processing and modeling
- **Pandas** – Data manipulation
- **Matplotlib** – Visualization
- **Scikit-learn** – Machine learning (Linear Regression)
- **Streamlit** – Interactive web dashboard

---

## 🔹 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/ridaafreen-er/Unemployment-Analysis.git
   cd Unemployment-Analysis
````

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux / Mac
   .venv\Scripts\activate     # Windows
   ````

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ````

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ````

5. Open the browser at:

   ```
   http://localhost:8501
   ````

---

## 🔹 How It Works

1. **Data Loading:** Reads `Unemployment in India.csv`
2. **Data Cleaning:** Removes nulls, converts types, strips whitespace
3. **Feature Engineering:** Extracts `Year` and `Month` from `Date`
4. **Visualization:** Plots unemployment trends over time
5. **Prediction:** Linear Regression model predicts unemployment for user-specified future date
6. **Evaluation:** MAE, RMSE, and R² Score for model performance

---

## 🔹 Project Insights

* Unemployment trends vary by region in India
* Linear regression captures general trends but may overestimate if predicting far into the future
* Predictions are more reliable **within the range of historical data**

---

## 🔹 Limitations

* Linear Regression may produce unrealistic values for long-term predictions
* Does not account for sudden economic events (e.g., COVID, policy changes)
* Future improvements could include **Random Forest, ARIMA, or LSTM models**

---

## 🔹 Future Scope

* Add multi-region comparison plots
* Deploy on **Streamlit Cloud** for public access
* Implement more advanced ML models for better predictions
* Add confidence intervals to forecasted values

---

## 🔹 Author

**Rida Aafreen**

* GitHub: [ridaafreen-er](https://github.com/ridaafreen-er)
* Data Science & Python Enthusiast

```






