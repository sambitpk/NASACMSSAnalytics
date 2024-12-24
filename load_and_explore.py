# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path, sep=' ', header=None)
    data = data.dropna(axis=1)  # Drop unnecessary columns
    columns = ['engine_id', 'cycle', 'operational_setting_1', 'operational_setting_2',
               'operational_setting_3', 'sensor_1', 'sensor_2', 'sensor_3', 
               'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 
               'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 
               'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18', 
               'sensor_19', 'sensor_20', 'sensor_21']
    data.columns = columns
    return data

# Display basic dataset info
def explore_data(data):
    print("Dataset Overview:")
    print(data.head())
    print("\nData Info:")
    print(data.info())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nSummary Statistics:")
    print(data.describe())

# Visualize sample trends
def visualize_data(data, engine_id, sensors):
    engine_data = data[data['engine_id'] == engine_id]
    plt.figure(figsize=(10, 6))
    for sensor in sensors:
        plt.plot(engine_data['cycle'], engine_data[sensor], label=sensor)
    plt.title(f'Sensor Readings for Engine {engine_id}')
    plt.xlabel('Cycle')
    plt.ylabel('Sensor Values')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = 'D:/LLM Learing/NASA TURBOFAN CASE STUDY/CMAPSSData/train_FD001.txt'  # Update with your file path
    data = load_data(file_path)
    explore_data(data)
    visualize_data(data, engine_id=1, sensors=['sensor_1', 'sensor_2', 'sensor_3'])


# Replace inf and -inf values with NaN in the entire DataFrame
data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
#####Exploratory Data Analysis (EDA)####
# Heatmap for feature correlations
correlation = data.iloc[:, 2:].corr()  # Exclude the first two columns (engine_id and cycle)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()

# Visualize sample trends for a specific engine
engine_data = data[data['engine_id'] == 1]  # Filter data for engine 1
plt.figure(figsize=(10, 6))
for sensor in ['sensor_1', 'sensor_2', 'sensor_3']:  # Choose sensors to visualize
    plt.plot(engine_data['cycle'], engine_data[sensor], label=sensor)
plt.title('Sensor Readings for Engine 1')
plt.xlabel('Cycle')
plt.ylabel('Sensor Values')
plt.legend()
plt.show()

## Feature Engineering ##
def create_lag_features(data, lag=1):
    for sensor in data.columns[5:]:  # Assuming sensor columns start from the 6th column
        data[f'{sensor}_lag_{lag}'] = data.groupby('engine_id')[sensor].shift(lag)
    return data

# Example: Create lag features for 3 previous cycles
data = create_lag_features(data, lag=1)
data = create_lag_features(data, lag=2)
data = create_lag_features(data, lag=3)

# Rolling AVerage
def create_rolling_features(data, window=3):
    for sensor in data.columns[5:]:
        data[f'{sensor}_rolling_mean'] = data.groupby('engine_id')[sensor].rolling(window).mean().reset_index(0, drop=True)
    return data

# Example: Create rolling features with a window of 3
data = create_rolling_features(data, window=3)





