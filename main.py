import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv", encoding='ISO-8859-1')

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=False)  # US format MM/DD/YYYY

# Group sales by date
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# Plot daily sales
plt.figure(figsize=(12,6))
plt.plot(daily_sales['Order Date'], daily_sales['Sales'], color='blue')
plt.title("Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.show()
# Extract time features
daily_sales['year'] = daily_sales['Order Date'].dt.year
daily_sales['month'] = daily_sales['Order Date'].dt.month
daily_sales['day'] = daily_sales['Order Date'].dt.day
daily_sales['day_of_week'] = daily_sales['Order Date'].dt.dayofweek
# Convert date to numeric
daily_sales['date_ordinal'] = daily_sales['Order Date'].map(pd.Timestamp.toordinal)
X = daily_sales[['date_ordinal']]
y = daily_sales['Sales']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
plt.figure(figsize=(12,6))

plt.plot(daily_sales['Order Date'], daily_sales['Sales'], label='Actual')
plt.plot(
    X_test['date_ordinal'].map(pd.Timestamp.fromordinal),
    y_pred,
    label='Forecast')

plt.title("Sales Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.savefig("sales_trend.png")
plt.savefig("forecast.png")
plt.show()
