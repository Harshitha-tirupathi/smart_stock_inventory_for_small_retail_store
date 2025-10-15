 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("retail_store_inventory.csv")
print("Data Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nColumn Info:\n", df.info())
print("\nBasic Statistics:\n", df.describe())

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print("\nMissing Values:\n", df.isnull().sum())
df = df.dropna(subset=['Date', 'Store ID', 'Product ID', 'Units Sold'])
numeric_columns = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing']
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

categorical_columns = ['Category', 'Region', 'Weather Condition', 'Seasonality']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

df = df.drop_duplicates()

df = df.sort_values(by=['Store ID', 'Product ID', 'Date'])

print("\nAfter Cleaning:")
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek
df['inventory_turnover'] = df['Units Sold'] / (df['Inventory Level'] + 1)  # +1 to avoid division by zero
df['forecast_accuracy'] = (df['Units Sold'] - df['Demand Forecast']).abs()
df['sell_through_rate'] = (df['Units Sold'] / (df['Inventory Level'] + df['Units Sold'])) * 100
df['units_sold_ma7'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df['lag_1_units_sold'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(1)
df['is_holiday'] = df['Holiday/Promotion'].apply(lambda x: 1 if x == 1 else 0)
df['has_discount'] = df['Discount'].apply(lambda x: 1 if x > 0 else 0)
df['price_vs_competitor'] = df['Price'] - df['Competitor Pricing']

print("\nFeature Engineering Completed")
print("New columns:", [col for col in df.columns if col not in ['Date', 'Store ID', 'Product ID', 'Category', 'Region', 
                                                           'Inventory Level', 'Units Sold', 'Units Ordered', 
                                                           'Demand Forecast', 'Price', 'Discount', 
                                                           'Weather Condition', 'Holiday/Promotion', 
                                                           'Competitor Pricing', 'Seasonality']])


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
daily_sales = df.groupby('Date')['Units Sold'].sum().reset_index()
plt.plot(daily_sales['Date'], daily_sales['Units Sold'])
plt.title('Daily Total Units Sold Trend')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.xticks(rotation=45)


plt.subplot(2, 2, 2)
category_sales = df.groupby('Category')['Units Sold'].sum().sort_values(ascending=False)
category_sales.plot(kind='bar')
plt.title('Total Units Sold by Category')
plt.xlabel('Category')
plt.ylabel('Units Sold')
plt.xticks(rotation=45)


plt.subplot(2, 2, 3)
region_sales = df.groupby('Region')['Units Sold'].sum().sort_values(ascending=False)
region_sales.plot(kind='bar')
plt.title('Total Units Sold by Region')
plt.xlabel('Region')
plt.ylabel('Units Sold')


plt.subplot(2, 2, 4)
store_sales = df.groupby('Store ID')['Units Sold'].sum().sort_values(ascending=False)
store_sales.plot(kind='bar')
plt.title('Total Units Sold by Store')
plt.xlabel('Store ID')
plt.ylabel('Units Sold')

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.histplot(df['Units Sold'], bins=30, kde=True)
plt.title('Distribution of Units Sold')

plt.subplot(2, 3, 2)
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Distribution of Prices')

plt.subplot(2, 3, 3)
sns.histplot(df['Inventory Level'], bins=30, kde=True)
plt.title('Distribution of Inventory Levels')

plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='Category', y='Units Sold')
plt.title('Units Sold by Category')
plt.xticks(rotation=45)

plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='Region', y='Units Sold')
plt.title('Units Sold by Region')

plt.subplot(2, 3, 6)
sns.scatterplot(data=df, x='Price', y='Units Sold', alpha=0.5)
plt.title('Price vs Units Sold')

plt.tight_layout()
plt.show()


top_products = df.groupby('Product ID')['Units Sold'].sum().nlargest(5).index.tolist()

plt.figure(figsize=(15, 10))
for i, product in enumerate(top_products[:3], 1):  # Show first 3 products
    plt.subplot(2, 2, i)
    product_data = df[df['Product ID'] == product]
    daily_product_sales = product_data.groupby('Date')['Units Sold'].sum().reset_index()
    
    plt.plot(daily_product_sales['Date'], daily_product_sales['Units Sold'], 
             label=f"Daily Sales - {product}", alpha=0.7)
    
    product_data_sorted = product_data.sort_values('Date')
    ma_7 = product_data_sorted.groupby('Date')['Units Sold'].sum().rolling(7, min_periods=1).mean()
    plt.plot(ma_7.index, ma_7.values, label="7-Day MA", linewidth=2)
    
    plt.title(f"Sales Trend - {product}")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
monthly_sales = df.groupby('month')['Units Sold'].sum().reset_index()
plt.plot(monthly_sales['month'], monthly_sales['Units Sold'], marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Units Sold')

plt.subplot(1, 2, 2)
season_sales = df.groupby('Seasonality')['Units Sold'].sum().reset_index()
plt.bar(season_sales['Seasonality'], season_sales['Units Sold'])
plt.title('Sales by Season')
plt.xlabel('Season')
plt.ylabel('Units Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
store_metrics = df.groupby('Store ID').agg({
    'Units Sold': 'sum',
    'Inventory Level': 'mean',
    'Price': 'mean',
    'Discount': 'mean'
}).reset_index()
store_metrics['efficiency'] = store_metrics['Units Sold'] / store_metrics['Inventory Level']
print("\nStore Performance Metrics:")
print(store_metrics.sort_values('efficiency', ascending=False))
df.to_csv("cleaned_retail_inventory_data.csv", index=False)
print("\nPreprocessed data saved as cleaned_retail_inventory_data.csv")
print(f"Final dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")