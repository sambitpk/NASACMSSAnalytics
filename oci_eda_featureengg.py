import oci

def fetch_from_oci(bucket_name, object_name, download_path):
    # Load OCI configuration
    config = oci.config.from_file("C:/Users/sambi/.oci/config.txt")  # Path to config file

    # Initialize Object Storage client
    object_storage = oci.object_storage.ObjectStorageClient(config)

    # Get namespace
    namespace = object_storage.get_namespace().data

    # Fetch the object
    try:
        print(f"Fetching {object_name} from bucket {bucket_name}...")
        obj = object_storage.get_object(namespace, bucket_name, object_name)
        
        # Save file locally
        with open(download_path, 'wb') as f:
            f.write(obj.data.content)
        print(f"File downloaded successfully to {download_path}.")
    except oci.exceptions.ServiceError as e:
        print(f"Error fetching file: {e}")

# Example Usage
bucket_name = "bucket-20241222-2100_DataanalysisTEST"
object_name = "train_FD001.txt"
download_path = "D:/LLM Learing/NASA TURBOFAN CASE STUDY/OCI/train"

fetch_from_oci(bucket_name, object_name, download_path)

# Load the dataset into a DataFrame
import pandas as pd

data = pd.read_csv(download_path, sep=' ', header=None)
data = data.dropna(axis=1)  # Drop unnecessary columns

# Add column names
columns = ['engine_id', 'cycle', 'operational_setting_1', 'operational_setting_2',
           'operational_setting_3', 'sensor_1', 'sensor_2', 'sensor_3', 
           'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 
           'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 
           'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18', 
           'sensor_19', 'sensor_20', 'sensor_21']
data.columns = columns

# Preview the dataset
print(data.head())

##EDA

# Dataset info
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Summary statistics
print(data.describe())
# Number of engines
print(f"Number of unique engines: {data['engine_id'].nunique()}")

# Distribution of cycles per engine
cycles_per_engine = data.groupby('engine_id')['cycle'].max()
print(cycles_per_engine.describe())

#Visualize Trends in Sensor Data

import matplotlib.pyplot as plt

# Plot sensor data for a single engine
engine_1 = data[data['engine_id'] == 1]

plt.figure(figsize=(10, 6))
for sensor in ['sensor_1', 'sensor_2', 'sensor_3']:
    plt.plot(engine_1['cycle'], engine_1[sensor], label=sensor)

plt.title('Sensor Readings for Engine 1')
plt.xlabel('Cycle')
plt.ylabel('Sensor Values')
plt.legend()
plt.show()

#Correlation analysis

import seaborn as sns

# Correlation matrix
correlation_matrix = data.iloc[:, 5:].corr()

# Heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Sensors')
plt.show()

###########Data Preprocessing (1. Calculate Remaining Useful Life (RUL) (Remaining Useful Life (RUL) is the target var))####
# Calculate RUL
data['RUL'] = data.groupby('engine_id')['cycle'].transform(max) - data['cycle']
print(data[['engine_id', 'cycle', 'RUL']].head())

##Normalize Sensor Data###
from sklearn.preprocessing import StandardScaler

# Select sensor columns
sensor_columns = [col for col in data.columns if 'sensor' in col]

# Apply standard scaling
scaler = StandardScaler()
data[sensor_columns] = scaler.fit_transform(data[sensor_columns])

print(data[sensor_columns].head())


### Feature Engineering #####
##Create Lag Features
##Lag features capture the historical values of a sensor to predict future outcomes.
# Create lag features for selected sensors
lag_features = ['sensor_1', 'sensor_2', 'sensor_3']
for sensor in lag_features:
    for lag in range(1, 4):  # Create lag-1, lag-2, lag-3
        data[f'{sensor}_lag{lag}'] = data.groupby('engine_id')[sensor].shift(lag)

# Display sample data with lag features
print(data[['engine_id', 'cycle', 'sensor_1', 'sensor_1_lag1', 'sensor_1_lag2']].head(10))


###Compute Rolling Averages
###Rolling averages smooth fluctuations and capture trends over time.
# Create rolling mean features for selected sensors
for sensor in lag_features:
    data[f'{sensor}_rolling_mean'] = data.groupby('engine_id')[sensor].rolling(window=5).mean().reset_index(0, drop=True)

# Display sample data with rolling mean
print(data[['engine_id', 'cycle', 'sensor_1', 'sensor_1_rolling_mean']].head(10))

## Id degradation in sesor features
# Calculate degradation indicators
# Calculate degradation indicators using transform
for sensor in lag_features:
    data[f'{sensor}_degradation'] = data.groupby('engine_id')[sensor].transform(lambda x: x - x.mean())

# Display sample data with degradation indicators
print(data[['engine_id', 'cycle', 'sensor_1', 'sensor_1_degradation']].head(10))

# Drop NaN values
data = data.dropna()

# Verify dataset size after cleaning
print(f"Data shape after cleaning: {data.shape}")
# Define feature columns and target
features = [col for col in data.columns if 'sensor' in col and 'degradation' not in col] + ['cycle']
target = 'RUL'

# Final dataset
X = data[features]
y = data[target]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

from sklearn.model_selection import train_test_split

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

#################MODEL BUILDING (BAseline Regression model) #######################################
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize the baseline model
baseline_model = LinearRegression()

# Train the model on training data
baseline_model.fit(X_train, y_train)

# Predict on the test set
y_pred_baseline = baseline_model.predict(X_test)

# Evaluate the baseline model
print("Baseline Model Performance:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_baseline):.2f}")
print(f"Root Mean Squared Error (RMSE): {mean_squared_error(y_test, y_pred_baseline, squared=False):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_baseline):.2f}")

#############ADVANCE MODEL (XGBOOST)#############################
from xgboost import XGBRegressor

# Initialize the XGBoost model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
print("XGBoost Model Performance:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_xgb):.2f}")
print(f"Root Mean Squared Error (RMSE): {mean_squared_error(y_test, y_pred_xgb, squared=False):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_xgb):.2f}")

##visualize ###

import matplotlib.pyplot as plt

# Plot true vs. predicted RUL
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label="True RUL", alpha=0.7)
plt.scatter(range(len(y_pred_xgb)), y_pred_xgb, label="Predicted RUL", alpha=0.7)
plt.title("True vs Predicted RUL")
plt.xlabel("Sample Index")
plt.ylabel("Remaining Useful Life")
plt.legend()
plt.show()
# Plot feature importance
importances = xgb_model.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(12, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

import pickle

# Save the model to a file
with open("xgb_rul_model.pkl", "wb") as model_file:
    pickle.dump(xgb_model, model_file)

print("Model saved successfully as xgb_rul_model.pkl")

