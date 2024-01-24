# -*- coding: utf-8 -*-

# -- Sheet --

# # Data


# ## SRBC Historical Dataset
# Load SRBC Dataset (360 CSV Files provisioned by SRBC Leadership Team)


import os
import pandas as pd

# Load, Merge, Create Pivoted Table for SRBC Historical Dataset (360 CSV Files)

# Path to the folder containing the CSV files
folder_path = r"C:\Users\mhriv\OneDrive\Desktop\Susquehanna River Basin Commission\PhD Dissertation\Water_Quality_Data\Data"

# List all files in the current directory (which includes the attachments in Datalore)
all_files = os.listdir()

# Filter to get only CSV files
csv_files = [file for file in all_files if file.endswith(".csv")]

# Create an empty DataFrame to store the merged data
merged_data = pd.DataFrame(columns=['Station_ID', 'Timestamp (UTC-05:00)', 'Parameter', 'File_Name', 'Value'])

# Iterate over each CSV file
for file in csv_files:
    # Extract the station ID and parameter from the file name
    station_id, parameter = os.path.splitext(file)[0].split('_')  # Assumes the file names have the format "station_id_parameter.csv"

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Add the station ID as a new column in the DataFrame
    df['Station_ID'] = station_id

    # Add the parameter as a new column in the DataFrame
    df['Parameter'] = parameter

    # Add the file name as a new column in the DataFrame
    df['File_Name'] = file

    # Merge the current file's data with the merged_data DataFrame
    merged_data = pd.concat([merged_data, df])

# Reset the index of the merged data
merged_data.reset_index(drop=True, inplace=True)
merged_data

# Pivot the Parameter column into separate columns
pivoted_data = merged_data.pivot(index=['Station_ID', 'Timestamp (UTC-05:00)'], columns='Parameter', values='Value').reset_index()
pivoted_data

# ## Data Pre-Processing
# - Extract Date/Time from the column 'Timestamp (UTC-05:00)'
# - Remove Time column since it does not have time


# Use errors='coerce' to handle errors
pivoted_data['Date'] = pd.to_datetime(pivoted_data['Timestamp (UTC-05:00)'], errors='coerce').dt.date
pivoted_data['Time'] = pd.to_datetime(pivoted_data['Timestamp (UTC-05:00)'], errors='coerce').dt.time

# Drop the original 'Timestamp (UTC-05:00)' column
pivoted_data = pivoted_data.drop(['Timestamp (UTC-05:00)'], axis=1)

pivoted_data = pivoted_data[['Station_ID','Date','DO', 'SpCond','Turb','WTemp','pH']]
pivoted_data

# Count the Number of Station_ID that will be considered in the Machine Learning Model
pivoted_data.Station_ID.value_counts()

# Drop the original 'Timestamp (UTC-05:00)' column
pivoted_data = pivoted_data.drop(['Station_ID','Date'], axis=1)
pivoted_data

# - Apply Inter-Quartile Range & Remove Outliers
# - According to the [United States Environmental Protection Agency (EPA)](https://archive.epa.gov/water/archive/web/html/vms59.html#:~:text=The%20conductivity%20of%20rivers%20in,150%20and%20500%20%C2%B5hos%2Fcm.), the conductivity of rivers in the United States generally ranges from *50 to 1500 µmhos/cm*. In addition, the studies of inland fresh waters indicate that streams supporting good mixed fisheries have a range between *150 and 500 µhos/cm.*


#Remove date and site , Find inter quartile range, Remove outlier outside that range
pivoted_data=pivoted_data[['DO', 'pH', 'Turb', 'SpCond', 'WTemp']]
Q1 = pivoted_data.quantile(0.25)
Q3= pivoted_data.quantile(0.75)
IQR = Q3 - Q1
pivoted_data = pivoted_data[~((pivoted_data < (Q1 - 1.5 * IQR)) |(pivoted_data > (Q3 + 1.5 * IQR))).any(axis=1)]
pivoted_data

#remove negative data observations for SpCond and set range starting from 50 µmhos/cm
pivoted_data = pivoted_data[pivoted_data['SpCond'] >= 50]
pivoted_data

# ## Feature Engineering


# ### **Ratio of Turbidity and Specific Conductance**
# 
# *  According to [EPA](https://www.epa.gov/national-aquatic-resource-surveys/indicators-conductivity), Conductivity is a measure of the ability of water to pass an electrical current. Since the dissolved salts and other inorganic chemicals conduct electrical current, conductivity increases as salinity increases.
# * According to [USGS](https://www.usgs.gov/special-topics/water-science-school/science/turbidity-and-water) Turbidity is the measure of relative clarity of a liquid. It is an optical characteristic of water and is a measurement of the amount of light that is scattered by material in the water when a light is shined through the water sample. Material that causes water to be turbid include clay, silt, very tiny inorganic and organic matter, algae, dissolved colored organic compounds, and plankton and other microscopic organisms.
# 
# * Assessing the ratio between Turbidity and Specific Conductance support evaluating the impact of the water to conduct electricity


import numpy as np
# Create the new feature
pivoted_data['Turb_to_SpCond'] = pivoted_data['Turb'] / pivoted_data['SpCond']

# Replace any infinities that could have been created by division by zero
pivoted_data['Turb_to_SpCond'].replace([np.inf, -np.inf], np.nan, inplace=True)
pivoted_data

# ### **Ratio between Dissolved Oxygen and Specific Conductance**
# 
# *  The water quality parameter of Dissolved Oxygen (DO) present an inverse relationship with the target variable Specific Conductance (SpCond)
# 
# * [Adina et al. (2016)](https://www.researchgate.net/publication/329574331_EVALUATION_OF_THE_ANTHROPOGENIC_IMPACT_ON_THE_WATER_QUALITY_OF_DAMBOVITA_RIVER) illustrates the correlation of both DO and SpCond highlighting its inverse proportionality due to the increase of Dissolved Oxygen that takes place when polluting substances that causes turbidity are eliminated from water.
# 
# * The ratio between Dissolved Oxygen and Specific Conductance may provide more insights of this inver relationship when Training Machine Learning Model


import numpy as np

# Create the new feature with inverse proportion
pivoted_data['DO_to_SpCond'] = pivoted_data['DO'] / pivoted_data['SpCond']

# Replace any infinities that could have been created by division by zero
pivoted_data['DO_to_SpCond'].replace([np.inf, -np.inf], np.nan, inplace=True)
pivoted_data

# ### **Ratio between Water Temperature and Specific Conductance**
# 
# *  The water quality Water Temperature (WTemp) impacts the variable Specific Conductance (SpCond) with levels of water acidity
# 
# * According to the [United States Environmental Protection Agency (EPA)](https://archive.epa.gov/water/archive/web/html/vms59.html) Conductivity is affected by Water Temperature. For instance, the warmer the water, the higher the conductivity. For this reason, Conductivity is reported as Conductivity at 25 degrees Celsius (25 C)
# 
# * [Hong et al. (2021)](https://doi.org/10.3390/environments8010006) Conductivity of water (and by extension dissolved solids) is expected to increase with temperature. For each 1 °C increment, conductivity rises by 2–4%. Temperature influences conductivity by increasing ion mobility and dissolvability of many salts and minerals


import numpy as np
# Create the new feature
pivoted_data['WTemp_to_SpCond'] = pivoted_data['WTemp'] / pivoted_data['SpCond']

# Replace any infinities that could have been created by division by zero
pivoted_data['WTemp_to_SpCond'].replace([np.inf, -np.inf], np.nan, inplace=True)
pivoted_data

# ### **Ratio between pH and Specific Conductance**
# 
# * According to the [United States Environmental Protection Agency (EPA)](https://www.epa.gov/system/files/documents/2021-07/parameter-factsheet_ph.pdf) pH is an important water quality parameter and indicator of chemical, physical, and biological changes in a waterbody. It plays a critical role in chemical processes in natural waters.
# 
# * [Saalidong et al. (2022)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262117) concluded the non-robustness of predictors in the surface water models were conspicuous.The authors state the relationship between water pH and other water quality parameters are different in different open water systems and can be influenced by the presence of other parameters. Therefore, the ratio of pH to SpCond allow reading underlying pattern of data.


import numpy as np
# Create the new feature
pivoted_data['pH_to_SpCond'] = pivoted_data['pH'] / pivoted_data['SpCond']

# Replace any infinities that could have been created by division by zero
pivoted_data['pH_to_SpCond'].replace([np.inf, -np.inf], np.nan, inplace=True)
pivoted_data

# ## Data Imputation


# * Apply Data Imputation technique to fill missing values using the method of Piecewise Multivariate Imputation
# * [Chen et al (2023)](https://doi.org/10.1016/j.jhydrol.2022.128901) proposes Multivariate Imputation such as Piece-wise Imputation (PWIMP)
#  * PWIMP divides the original dataset into a few sub-datasets based on the missing segments, and then each sub-dataset is filled in and combined to obtain a complete dataset.
#  * PWIMP can reflect both of the uncertainty and temporal information of the original data.


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Define the imputer
imp = IterativeImputer(max_iter=5, random_state=42)

# Define index cut-off for demonstration purposes
# You might have a more complex criteria based on your data
index_cut_off = len(pivoted_data) // 4

# Split the data into two segments
segment1 = pivoted_data.iloc[:index_cut_off, :]
segment2 = pivoted_data.iloc[index_cut_off:, :]

# Apply the imputer on each segment
segment1_imputed = imp.fit_transform(segment1)
segment2_imputed = imp.fit_transform(segment2)

# Convert imputed segments back to DataFrame
segment1_imputed = pd.DataFrame(segment1_imputed, columns=pivoted_data.columns, index=segment1.index)
segment2_imputed = pd.DataFrame(segment2_imputed, columns=pivoted_data.columns, index=segment2.index)

# Merge imputed segments
imputed_data = pd.concat([segment1_imputed, segment2_imputed])
pivoted_data = imputed_data
pivoted_data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the Seaborn style
sns.set_style("white")

# 1. Bar chart showing the number of missing values per feature before imputation
missing_values_before = segment1.isnull().sum() + segment2.isnull().sum()

plt.figure(figsize=(12, 7))
sns.barplot(x=missing_values_before.index, y=missing_values_before.values, palette="viridis", edgecolor='black')
plt.title('Number of Missing Values per Feature Before Multivariate Piecewise Imputation (PWIMP)', fontsize=18, fontweight='bold')
plt.ylabel('Number of Missing Values', fontsize=15, fontweight='bold')
plt.xlabel('Features', fontsize=15, fontweight='bold')
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# 2. Parallel bar chart to display the number of imputations performed per feature for each segment

# Calculate the number of imputed values per segment using the DataFrame representations
imputed_values_segment1 = segment1.isnull().sum() - segment1_imputed.isnull().sum()
imputed_values_segment2 = segment2.isnull().sum() - segment2_imputed.isnull().sum()

# Parallel bar chart
width = 0.35
ind = np.arange(len(imputed_values_segment1))
colors = sns.color_palette("viridis", n_colors=2)

plt.figure(figsize=(12, 7))
plt.bar(ind, imputed_values_segment1, width, color=colors[0], edgecolor='black', label='Segment 1 Imputations')
plt.bar(ind + width, imputed_values_segment2, width, color=colors[1], edgecolor='black', label='Segment 2 Imputations')

plt.title('Number of Imputations per Feature for Each Segment - Multivariate Piecewise Imputation (PWIMP)', fontsize=18, fontweight='bold')
plt.ylabel('Number of Imputations', fontsize=15, fontweight='bold')
plt.xlabel('Features', fontsize=15, fontweight='bold')
plt.xticks(ind + width / 2, imputed_values_segment1.index, rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

#Imputed Values for Segment 1
imputed_values_segment1

#Imputed Values for Segment 2
imputed_values_segment2

# Check Missing Values
pivoted_data.isnull().sum()

# ## Data Augmentation 


# *   According to [Iwana et al. (2021)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841#pone.0254841.ref020) Data augmentation technique attempts to increase the Generalization ability of Trained Models by reducing Overfitting and expanding the decision boundary of the model.
# *   [El Bilali et al. (2021)](https://www.sciencedirect.com/science/article/abs/pii/S0022169421005576?via%3Dihub) study applies Gaussian noises to the input variables generating 600 virtual dataset for enhancing the Training process to improve the Prediction accuracy of the study. The study area is the downstream part of the Bouregreg watershed and is located in the northwest of Morocco.
# * [Xu et al. (2020a, 2020b)](https://www.sciencedirect.com/science/article/abs/pii/S0043135420306400?via%3Dihub) created 20,000 virtual samples and applied Deep Neural Network Algorithm (DNN) including 16 layers, and 100 neurons in hidden layers and applied to Wastewater treatment.
# 
# This study applies Gaussian Noise Technique to create *Synthetic Samples* of the water quality parameters of the Susquehanna River Basin
# Synthetic data is generated with 50000 synthetic data set to a Standard Deviation of 0.3


import numpy as np
import pandas as pd

data = pivoted_data

# Specify the standard deviation for the Gaussian noise
std_dev = 0.5  # Adjust the standard deviation as desired

# Generate synthetic data using Gaussian noise
synthetic_data = pd.DataFrame()
num_samples = 20000 # Generate synthetic data with the same length as the original data

for column in data.columns[2:]:
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(data[column]):
        # Get the mean and standard deviation of the original data
        mean = data[column].mean()
        std = data[column].std()

        if np.isscalar(mean) and np.isscalar(std):
            # Generate synthetic data points based on the Gaussian distribution
            synthetic_values = np.random.normal(loc=mean, scale=std_dev * std, size=num_samples)

            # Add the synthetic data to the DataFrame
            synthetic_data[column] = synthetic_values

# Reset index for both DataFrames before concatenation
data.reset_index(drop=True, inplace=True)
synthetic_data.reset_index(drop=True, inplace=True)

# Concatenate the original and synthetic data
combined_data = pd.concat([data, synthetic_data], ignore_index=True)

# Verify the length of the combined data
original_length = len(data)
combined_length = len(combined_data)
print("Original length:", original_length)
print("Combined length:", combined_length)

# Fill missing values in the combined data with the mean of the respective column
combined_data = combined_data.fillna(combined_data.mean())

# Verify if there are any missing values in the combined data
if combined_data.isnull().sum().sum() > 0:
    print("Missing values still exist in the combined data.")
else:
    print("No missing values in the combined data.")

# Print the combined data
combined_data

# ## Pearson Correlation Assessment
# - Correlation assessment between original dataset and resulting 'combined_data' for Machine Learning Modeling


# ### Original Dataset


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your dataset is named 'pivoted_data' and contains the columns 'DO', 'pH', 'Turb', 'SpCond', 'WTemp'
parameters = ['SpCond','Turb','DO','WTemp','pH']
corr_matrix = pivoted_data[parameters].corr()

# Generate the heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, 
            cbar_kws={"ticks":[-1, -0.5, 0, 0.5, 1]}, 
            annot_kws={"size": 14, "weight": "bold"}, 
            fmt=".2f", linewidths=0.5, linecolor='black')

# Title and formatting
plt.title("Pearson Correlation Heatmap (Raw Data)", fontsize=18, fontweight='bold')
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)
plt.tight_layout()

# Adjust the linewidth of the spines to make them less bold
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.show()

# ### Pre-processed Dataset


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your dataset is named 'pivoted_data' and contains the columns 'DO', 'pH', 'Turb', 'SpCond', 'WTemp'
parameters = ['DO','pH','Turb','SpCond','WTemp','Turb_to_SpCond','DO_to_SpCond','WTemp_to_SpCond','pH_to_SpCond']
corr_matrix = combined_data[parameters].corr()

# Generate the heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, 
            cbar_kws={"ticks":[-1, -0.5, 0, 0.5, 1]}, 
            annot_kws={"size": 14, "weight": "bold"}, 
            fmt=".2f", linewidths=0.5, linecolor='black')

# Title and formatting
plt.title("Pearson Correlation Heatmap (Pre-Processed Data)", fontsize=18, fontweight='bold')
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)
plt.tight_layout()

# Adjust the linewidth of the spines to make them less bold
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.show()

# ## Descriptive Statistics for Specific Conductance (SpCond)
# - Scale assessment for Specific Conductance as Target Variable


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Remove negative data observations for SpCond
combined_data = combined_data[combined_data['SpCond'] >= 50]

# Set a style and palette with seaborn
sns.set_style('white')
palette = sns.color_palette('cool')

# Assuming combined_data is your DataFrame and "SpCond" is your target variable
SpCond = combined_data['SpCond']

print(SpCond.describe())

# Extract the first color from the viridis palette
color = sns.color_palette("viridis", n_colors=1)[0]

# Histogram
plt.figure(figsize=(12, 7))
sns.histplot(SpCond, bins=30, color=color, edgecolor='black')
plt.title('Histogram of Specific Conductance (SpCond)', fontsize=18, fontweight='bold')
plt.xlabel('Specific Conductance (µS/cm)', fontsize=15, fontweight='bold')
plt.ylabel('Frequency', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(12, 7))
sns.boxplot(x=SpCond, color=color)
plt.title('Box Plot of Specific Conductance (SpCond)', fontsize=20, fontweight='bold')
plt.xlabel('Specific Conductance (µS/cm)', fontsize=17, fontweight='bold')
plt.xticks(fontsize=15)
plt.tight_layout()
plt.show()

# ## Variance Inflation Factor (VIF)
# 
# *   According to [Fernandes et al. (2023)](https://doi.org/10.1016/j.mex.2023.102153) this study conducted a water quality prediction using environmental dataset to predict surface water quality parameters based on landscape metrics and contaminant loads.
# 
# *   The Authors introduced pre-processing techniques such as statistical significance of the estimations, **multicollinearity**, error normality, and homoscedasticity.
# 
# *    To measure the degree of multicollinearity is commonly calculated the **regressors Variance Inflation Factor (VIF)**. This factor ranges from 1 to infinite (when there is a perfect correlation between regressors) and must be lower as possible. Still, there are different recommendations for the ***maximum acceptable VIF threshold of 10, 7.5 or 5.***


pip install statsmodels --upgrade pip

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Filter combined_data to only have the relevant columns
df = combined_data[['DO', 'pH', 'Turb', 'SpCond', 'WTemp', 'Turb_to_SpCond', 'DO_to_SpCond', 'WTemp_to_SpCond', 'pH_to_SpCond']]

# Add a constant column for the intercept in the regression model
df = add_constant(df)

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = df.columns[1:]  # Exclude the constant column
vif_data["VIF"] = [variance_inflation_factor(df.values, i+1) for i in range(df.shape[1]-1)]  # Exclude the constant column

print(vif_data)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from matplotlib.lines import Line2D

# Filter combined_data to only have the relevant columns
df = combined_data[['DO', 'pH', 'Turb', 'SpCond', 'WTemp', 'Turb_to_SpCond', 'DO_to_SpCond', 'WTemp_to_SpCond', 'pH_to_SpCond']]

# Add a constant column for the intercept in the regression model
df = add_constant(df)

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = df.columns[1:]  # Exclude the constant column
vif_data["VIF"] = [variance_inflation_factor(df.values, i+1) for i in range(df.shape[1]-1)]  # Exclude the constant column

# Plot VIF values using Seaborn with the viridis palette
plt.figure(figsize=(12, 7))
sns.barplot(x="Feature", y="VIF", data=vif_data, palette="viridis", edgecolor='black')
line = plt.axhline(y=10, color='red', linestyle='--')
plt.title('VIF values for each feature', fontsize=18, fontweight='bold')
plt.xlabel('Feature', fontsize=15, fontweight='bold')
plt.ylabel('VIF', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)

# Create a custom legend
legend_line = Line2D([0], [0], color='red', linestyle='--', label='VIF Threshold = 10')
plt.legend(handles=[legend_line], loc='upper left', prop={'size': 14, 'weight':'bold'}, bbox_to_anchor=(0, 1.02))
plt.tight_layout()

# Adjust the linewidth of the spines to make them less bold
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.show()

# Select Dataframe for Machine Learning Modeling
combined_data = combined_data[['DO','pH', 'Turb', 'SpCond', 'WTemp','Turb_to_SpCond','WTemp_to_SpCond']]
combined_data

# Remove negative data observations for SpCond
combined_data = combined_data[combined_data['SpCond'] >= 50]
combined_data

# ## Machine Learning Techniques 


# ### Artificial Neural Networks (ANNs)
# - Standard ANNs


from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# First, separate the features and target variable
X = combined_data.drop(columns='SpCond')
y = combined_data['SpCond']

# Define the number of splits/folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define the fold number
fold_no = 1

# Perform cross-validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Reshape y to be a 2D array
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # Define the model structure
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))  # Input layer
    model.add(Dense(100, activation='relu'))  # First hidden layer
    model.add(Dense(100, activation='relu'))  # Second hidden layer
    model.add(Dense(100, activation='relu'))  # Third hidden layer
    model.add(Dense(1, activation='linear'))  # Output layer

    # Compile the model
    model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32, verbose=0)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print metrics
    print(f"Fold {fold_no} - R2 Score:", r2_score(y_test, y_pred))
    print(f"Fold {fold_no} - Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print(f"Fold {fold_no} - Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"Fold {fold_no} - Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("---------------------------------------")

    # Increment fold number
    fold_no += 1

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# First, separate the features and target variable
X = combined_data.drop(columns='SpCond')
y = combined_data['SpCond']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape y to be a 2D array
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Define the model structure
model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(Dense(100, activation='relu'))  # First hidden layer
model.add(Dense(100, activation='relu'))  # Second hidden layer
model.add(Dense(100, activation='relu'))  # Third hidden layer
model.add(Dense(1, activation='linear'))  # Output layer

# Compile the model
model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print metrics
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class ANNWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        pass  # The model is already trained

    def predict(self, X):
        return self.model.predict(X).flatten()

# Wrap the trained Keras model
wrapped_model = ANNWrapper(model)

# Use permutation importance to calculate the importance of each feature
result = permutation_importance(wrapped_model, X_test, y_test, scoring='r2', n_repeats=25, random_state=42)

# Sort features based on importance
sorted_idx = result.importances_mean.argsort()

# Print feature importances in a tabular format
print("Feature Importances:")
for i in sorted_idx[::-1]:
    print(f"{X_test.columns[i]:<25}: {result.importances_mean[i]:.4f} ± {result.importances_std[i]:.4f}")

# Set the Seaborn style
sns.set_style("white")

# Create a new figure with specified size
plt.figure(figsize=(12, 7))

# Create the bar plot with the "viridis" color palette and edge colors
sns.barplot(x=result.importances_mean[sorted_idx], y=X_test.columns[sorted_idx], 
            palette="viridis", edgecolor='black')

# Title, labels, and formatting
plt.title('Feature Importance - Artificial Neural Networks (ANNs)', 
          fontsize=18, fontweight='bold')
plt.xlabel('Permutation Importance', fontsize=15, fontweight='bold')
plt.ylabel('Feature', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adjust the linewidth of the spines to make them less bold
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# Ensure everything fits in the plot area
plt.tight_layout()

# Display the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Compute absolute errors
errors = np.abs(y_test - y_pred)

# Define a threshold for small vs. large error
threshold = np.percentile(errors, 50)  # Using the median as a threshold

# Determine colors and alphas based on error magnitude
point_colors = ['green' if e <= threshold else 'indigo' for e in errors]
alphas = [0.9 if e <= threshold else 0.5 for e in errors]

# Create a figure with a specified size
plt.figure(figsize=(12, 7))

# Scatter plot of actual vs. predicted values with customized colors and alphas
plt.scatter(y_test, y_pred, c=point_colors, alpha=alphas, edgecolor='black')

# Plot the ideal line (y=x line)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2)

# Additional plot decorations
plt.title('Actual vs Predicted Scatter Plot - Artificial Neural Networks (ANNs)', fontsize=18, fontweight='bold')
plt.xlabel('Actual Values', fontsize=15, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend to explain the colors
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Small Error', markersize=10, markerfacecolor='green', markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='w', label='Large Error', markersize=10, markerfacecolor='indigo', markeredgecolor='black')]
plt.legend(handles=legend_elements, fontsize=14)

plt.tight_layout()
plt.show()

# ### Long Short-Term Memory (LSTM)
# - Features as Timesteps


from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# First, separate the features and target variable
X = combined_data.drop(columns='SpCond')
y = combined_data['SpCond']

# Define the number of splits/folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define the fold number
fold_no = 1

# Perform cross-validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Reshape data for LSTM, treating each feature as a time step
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # Define the LSTM model structure
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32, verbose=0)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print metrics
    print(f"Fold {fold_no} - R2 Score:", r2_score(y_test, y_pred))
    print(f"Fold {fold_no} - Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print(f"Fold {fold_no} - Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"Fold {fold_no} - Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("---------------------------------------")

    # Increment fold number
    fold_no += 1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split

# First, separate the features and target variable
X = combined_data.drop(columns='SpCond')
y = combined_data['SpCond']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM, treating each feature as a time step
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Define the LSTM model structure
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(100))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print metrics
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class LSTMWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        pass  # The model is already trained

    def predict(self, X):
        # Reshape the data as expected by the LSTM
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X_reshaped).flatten()

# Before reshaping or splitting, save the feature names:
feature_names = combined_data.columns.drop('SpCond').to_numpy()

# Ensure X_test is a 2D array when passed into permutation_importance
X_test_2D = X_test
if len(X_test_2D.shape) == 3:
    X_test_2D = X_test_2D.reshape(X_test_2D.shape[0], -1)

# Wrap the trained Keras model
wrapped_model = LSTMWrapper(model)

# Use permutation importance to calculate the importance of each feature
result = permutation_importance(wrapped_model, X_test_2D, y_test, scoring='r2', n_repeats=25, random_state=42)

# Sort features based on importance
sorted_idx = result.importances_mean.argsort()

# Print feature importances in a tabular format
print("Feature Importances:")
for i in sorted_idx[::-1]:
    print(f"{feature_names[i]:<25}: {result.importances_mean[i]:.4f} ± {result.importances_std[i]:.4f}")

# Plotting
plt.figure(figsize=(12, 7))
sns.barplot(x=result.importances_mean[sorted_idx], y=feature_names[sorted_idx], palette="viridis")

# Styling
plt.title('Feature Importance - Long Short-Term Memory (LSTM)', fontsize=18, fontweight='bold')
plt.xlabel('Permutation Importance', fontsize=15, fontweight='bold')
plt.ylabel('Feature', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Compute absolute errors
errors = np.abs(y_test - y_pred)

# Define a threshold for small vs. large error
threshold = np.percentile(errors, 50)  # Using the median as a threshold

# Determine colors and alphas based on error magnitude
point_colors = ['green' if e <= threshold else 'indigo' for e in errors]
alphas = [0.9 if e <= threshold else 0.5 for e in errors]

# Create a figure with a specified size
plt.figure(figsize=(12, 7))

# Scatter plot of actual vs. predicted values with customized colors and alphas
plt.scatter(y_test, y_pred, c=point_colors, alpha=alphas, edgecolor='black')

# Plot the ideal line (y=x line)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2)

# Additional plot decorations
plt.title('Actual vs Predicted Scatter Plot - Long Short-Term Memory (LSTM)', fontsize=18, fontweight='bold')
plt.xlabel('Actual Values', fontsize=15, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend to explain the colors
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Small Error', markersize=10, markerfacecolor='green', markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='w', label='Large Error', markersize=10, markerfacecolor='indigo', markeredgecolor='black')]
plt.legend(handles=legend_elements, fontsize=14)

plt.tight_layout()
plt.show()

# ## Ensemble Methods


# ### Random Forest Algorithm
# - Standard Random Forest Algorithm


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to hold cross-validation results
r2_scores = []
mse_scores = []
rmse_scores = []
mae_scores = []

# Loop over each fold
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Train a Random Forest Regressor
    reg_fold = RandomForestRegressor(random_state=42)
    reg_fold.fit(X_train_fold, y_train_fold)

    # Make predictions on the test set
    y_pred_fold = reg_fold.predict(X_test_fold)

    # Calculate metrics
    r2 = r2_score(y_test_fold, y_pred_fold)
    mse = mean_squared_error(y_test_fold, y_pred_fold)
    rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
    mae = mean_absolute_error(y_test_fold, y_pred_fold)

    # Append to results
    r2_scores.append(r2)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    mae_scores.append(mae)

    # Print fold results
    print(f"Fold {fold}:")
    print("  R2 Score:", r2)
    print("  Mean Squared Error:", mse)
    print("  Root Mean Squared Error:", rmse)
    print("  Mean Absolute Error:", mae)
    print()

# Print average results
print("Mean R2 Score:", np.mean(r2_scores))
print("Mean Mean Squared Error:", np.mean(mse_scores))
print("Mean Root Mean Squared Error:", np.mean(rmse_scores))
print("Mean Mean Absolute Error:", np.mean(mae_scores))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.special import inv_boxcox
import numpy as np

# Separate the features and target variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Calculate and print metrics using the inverse-transformed values
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Separate the features and target variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Calculate Metrics
metrics = {
    "R2 Score": r2_score(y_test, y_pred),
    "Mean Squared Error": mean_squared_error(y_test, y_pred),
    "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
    "Mean Absolute Error": mean_absolute_error(y_test, y_pred)
}

# Display Metrics
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# Extract Feature Importances
feature_importances = reg.feature_importances_

# Convert to DataFrame for Visualization
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort Features by Importance
features_df = features_df.sort_values(by='Importance', ascending=False)
print(features_df)

# Plot Feature Importances using Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(y='Feature', x='Importance', data=features_df, palette='viridis')

# Styling and Decorations
plt.title('Feature Importance - Random Forest', fontsize=18, fontweight='bold')
plt.xlabel('Importance', fontsize=15, fontweight='bold')
plt.ylabel('Feature', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Compute absolute errors
errors = np.abs(y_test - y_pred)

# Define a threshold for small vs. large error
threshold = np.percentile(errors, 50)  # Using the median as a threshold

# Determine colors and alphas based on error magnitude
point_colors = ['green' if e <= threshold else 'indigo' for e in errors]
alphas = [0.9 if e <= threshold else 0.5 for e in errors]

# Create a figure with a specified size
plt.figure(figsize=(12, 7))

# Scatter plot of actual vs. predicted values with customized colors and alphas
plt.scatter(y_test, y_pred, c=point_colors, alpha=alphas, edgecolor='black')

# Plot the ideal line (y=x line)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2)

# Additional plot decorations
plt.title('Actual vs Predicted Scatter Plot - Random Forest', fontsize=18, fontweight='bold')
plt.xlabel('Actual Values', fontsize=15, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend to explain the colors
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Small Error', markersize=10, markerfacecolor='green', markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='w', label='Large Error', markersize=10, markerfacecolor='indigo', markeredgecolor='black')]
plt.legend(handles=legend_elements, fontsize=14)

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Assuming combined_data is your dataset and 'SpCond' is the target variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Fit the model
gbm.fit(X_train, y_train)

# Collect RMSE over each stage
train_rmse = []
validation_rmse = []
for stage_predictions in gbm.staged_predict(X_train):
    train_rmse.append(np.sqrt(mean_squared_error(y_train, stage_predictions)))

for stage_predictions in gbm.staged_predict(X_val):
    validation_rmse.append(np.sqrt(mean_squared_error(y_val, stage_predictions)))

# Convert the RMSE values to a DataFrame
data = pd.DataFrame({
    'Epoch': np.arange(1, 101),
    'Training RMSE': train_rmse,
    'Validation RMSE': validation_rmse
})

# Data Transformation for Plotting
data_melted = data.melt(
    id_vars=['Epoch'], 
    value_vars=['Training RMSE', 'Validation RMSE'],
    var_name='Dataset', 
    value_name='RMSE'
)

# Palette Configuration
colors = sns.color_palette("viridis", n_colors=2)
palette_dict = {'Training RMSE': colors[0], 'Validation RMSE': colors[1]}

# Plotting
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=data_melted, 
    x='Epoch', 
    y='RMSE', 
    hue='Dataset', 
    palette=palette_dict, 
    linewidth=2.5
)

# Plot Styling and Decorations
plt.title('RMSE Over Epochs - Random Forest', fontsize=18, fontweight='bold')
plt.xlabel('Epoch', fontsize=15, fontweight='bold')
plt.ylabel('RMSE', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

# Display Plot
plt.tight_layout()
plt.show()

# ### Decision Trees Algorithm


from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Separate the features and target variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Number of splits
n_splits = 5

# Initialize KFold cross-validator
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Loop through each fold
fold_num = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train a Decision Tree Regressor
    reg = DecisionTreeRegressor(random_state=42)
    reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = reg.predict(X_test)

    # Calculate and print metrics
    print(f"Fold {fold_num} Results:")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("============================")

    fold_num += 1

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# Separate the features and target variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Calculate and print metrics
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Separate Features and Target Variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)

# Predict on Test Set
y_pred = reg.predict(X_test)

# Calculate Metrics
metrics = {
    "R2 Score": r2_score(y_test, y_pred),
    "Mean Squared Error": mean_squared_error(y_test, y_pred),
    "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
    "Mean Absolute Error": mean_absolute_error(y_test, y_pred)
}

# Display Metrics
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# Extract Feature Importances
feature_importances = reg.feature_importances_

# Convert to DataFrame for Visualization
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort Features by Importance
features_df = features_df.sort_values(by='Importance', ascending=False)
print(features_df)

# Plot Feature Importances using Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(y='Feature', x='Importance', data=features_df, palette='viridis')

# Styling and Decorations
plt.title('Feature Importance - Decision Tree', fontsize=18, fontweight='bold')
plt.xlabel('Importance', fontsize=15, fontweight='bold')
plt.ylabel('Feature', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Separate Features and Target Variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)

# Predict on Test Set
y_pred = reg.predict(X_test)

# Compute absolute errors
errors = np.abs(y_test - y_pred)

# Define a threshold for small vs. large error
threshold = np.percentile(errors, 50)  # Using the median as a threshold

# Determine colors and alphas based on error magnitude
point_colors = ['green' if e <= threshold else 'indigo' for e in errors]
alphas = [0.9 if e <= threshold else 0.5 for e in errors]

# Create a figure with a specified size
plt.figure(figsize=(12, 7))

# Scatter plot of actual vs. predicted values with customized colors and alphas
plt.scatter(y_test, y_pred, c=point_colors, alpha=alphas, edgecolor='black')

# Plot the ideal line (y=x line)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2)

# Additional plot decorations
plt.title('Actual vs Predicted Scatter Plot - Decision Tree', fontsize=18, fontweight='bold')
plt.xlabel('Actual Values', fontsize=15, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend to explain the colors
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Small Error', markersize=10, markerfacecolor='green', markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='w', label='Large Error', markersize=10, markerfacecolor='indigo', markeredgecolor='black')]
plt.legend(handles=legend_elements, fontsize=14)

plt.tight_layout()
plt.show()

# ### Gradient Boosting Algorithm


from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Separate the features and target variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Number of splits
n_splits = 5

# Initialize KFold cross-validator
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Loop through each fold
fold_num = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train a Gradient Boosting Regressor
    reg = GradientBoostingRegressor(random_state=42)
    reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = reg.predict(X_test)

    # Calculate and print metrics
    print(f"Fold {fold_num} Results:")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("============================")

    fold_num += 1

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# Separate the features and target variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Regressor
reg = GradientBoostingRegressor(random_state=42)
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Calculate and print metrics
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Separate the features and target variable
X = combined_data.drop(['SpCond'], axis=1)
y = combined_data['SpCond']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Regressor
reg = GradientBoostingRegressor(random_state=42)
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Calculate Metrics
metrics = {
    "R2 Score": r2_score(y_test, y_pred),
    "Mean Squared Error": mean_squared_error(y_test, y_pred),
    "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
    "Mean Absolute Error": mean_absolute_error(y_test, y_pred)
}

# Display Metrics
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# Extract Feature Importances
feature_importances = reg.feature_importances_

# Convert to DataFrame for Visualization
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort Features by Importance
features_df = features_df.sort_values(by='Importance', ascending=False)
print(features_df)

# Plot Feature Importances using Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(y='Feature', x='Importance', data=features_df, palette='viridis')

# Styling and Decorations
plt.title('Feature Importance - Gradient Boosting', fontsize=18, fontweight='bold')
plt.xlabel('Importance', fontsize=15, fontweight='bold')
plt.ylabel('Feature', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Compute absolute errors
errors = np.abs(y_test - y_pred)

# Define a threshold for small vs. large error
threshold = np.percentile(errors, 50)  # Using the median as a threshold

# Determine colors and alphas based on error magnitude
point_colors = ['green' if e <= threshold else 'indigo' for e in errors]
alphas = [0.9 if e <= threshold else 0.5 for e in errors]

# Create a figure with a specified size
plt.figure(figsize=(12, 7))

# Scatter plot of actual vs. predicted values with customized colors and alphas
plt.scatter(y_test, y_pred, c=point_colors, alpha=alphas, edgecolor='black')

# Plot the ideal line (y=x line)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2)

# Additional plot decorations
plt.title('Actual vs Predicted Scatter Plot - Gradient Boosting', fontsize=18, fontweight='bold')
plt.xlabel('Actual Values', fontsize=15, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=15, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend to explain the colors
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Small Error', markersize=10, markerfacecolor='green', markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='w', label='Large Error', markersize=10, markerfacecolor='indigo', markeredgecolor='black')]
plt.legend(handles=legend_elements, fontsize=14)

plt.tight_layout()
plt.show()

