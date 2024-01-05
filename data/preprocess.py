import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './bank-full.csv'
data = pd.read_csv(file_path, delimiter=';')

# List of all the columns in the dataset
columns = data.columns

# # Number of rows and columns for the subplot grid
# # Adjusting the rows to fit all features
# n_cols = 4
# n_rows = (len(columns) + n_cols - 1) // n_cols  # Ceil division to fit all features

# # Creating subplots
# fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
# fig.suptitle('Feature Distribution in Dataset', fontsize=16)

# # Iterate over the columns and plot the distributions
# for i, col in enumerate(columns):
#     row = i // n_cols
#     col = i % n_cols
#     if data[columns[i]].dtype == 'object':
#         # For categorical data, use count plot
#         sns.countplot(x=data[columns[i]], ax=axs[row, col])
#     else:
#         # For numerical data, use histogram
#         sns.histplot(data[columns[i]], kde=True, ax=axs[row, col])
    
#     axs[row, col].set_title(columns[i])
#     axs[row, col].set_xlabel('')
#     axs[row, col].set_ylabel('')

# # Hide unused subplots
# for i in range(len(columns), n_rows * n_cols):
#     axs[i // n_cols, i % n_cols].axis('off')

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.subplots_adjust(top=0.95)

# # Save the plot
# output_file = './feature_distribution.png'
# plt.savefig(output_file)


from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Step 2: Convert binary features to 0 or 1
binary_features = ['default', 'housing', 'loan', 'y']
data[binary_features] = data[binary_features].applymap(lambda x: 1 if x == 'yes' else 0)

# Step 3: Convert categorical features to one-hot encoding
categorical_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
data = pd.get_dummies(data, columns=categorical_features, prefix=categorical_features)

# Step 4: Z-score normalization for numeric features
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])


# Calculating the correlation matrix
correlation_matrix = data.corr()

# # Plotting the correlation matrix
# plt.figure(figsize=(20, 15))
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
# plt.title('Feature Correlation Matrix')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)

# # Save the correlation matrix plot
# correlation_file = './feature_correlation.png'
# plt.savefig(correlation_file)


# Finding features with correlation greater than 0.9 (excluding self-correlation)
high_corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_features.add(correlation_matrix.columns[i])
            print(f"High correlation ({correlation_matrix.iloc[i, j]:.2f}) between '{correlation_matrix.columns[i]}' and '{correlation_matrix.columns[j]}'")

# Removing one of each pair of highly correlated features
data_reduced = data.drop(columns=high_corr_features)

# Moving the label 'y' to the last column
label = data_reduced.pop('y')
data_reduced['y'] = label


from sklearn.model_selection import train_test_split
import os
# Splitting the dataset into train, dev, and test sets with ratios 8:1:1
train_set, temp_set = train_test_split(data_reduced, test_size=0.2, random_state=42)
dev_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

# Saving the datasets to CSV files
os.makedirs("split", exist_ok=True)
train_file = 'split/train.csv'
dev_file = 'split/dev.csv'
test_file = 'split/test.csv'

train_set.to_csv(train_file, index=False)
dev_set.to_csv(dev_file, index=False)
test_set.to_csv(test_file, index=False)
