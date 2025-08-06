# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 📌 Step 2: Load Dataset
file_path = file_path = r"C:\Users\HP\Downloads\MentalHealthSurvey.xlsx"
  # Change if needed
df = pd.read_excel(file_path)

# 📌 Step 3: Show Basic Info
print("📊 Shape:", df.shape)
print("🧠 Columns:", df.columns.tolist())
print("📋 Data Types:\n", df.dtypes)
print("❓ Missing Values:\n", df.isnull().sum())
print("🔎 First Rows:\n", df.head())

# 📌 Step 4: Convert Range Columns
def convert_range_to_avg(val):
    try:
        if isinstance(val, str) and '-' in val:
            low, high = map(float, val.split('-'))
            return (low + high) / 2
        return float(val)
    except:
        return np.nan

df['cgpa'] = df['cgpa'].apply(convert_range_to_avg)
df['average_sleep'] = df['average_sleep'].apply(convert_range_to_avg)

# 📌 Step 5: Binary Conversion for Depression & Anxiety
df['depression'] = df['depression'].apply(lambda x: 1 if x >= 3 else 0)
df['anxiety'] = df['anxiety'].apply(lambda x: 1 if x >= 3 else 0)

# 📌 Step 6: Fill Missing Values
df.fillna(df.mode().iloc[0], inplace=True)

# 📌 Step 7: Encode Categorical Variables
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('stress_relief_activities')  # we'll handle this separately
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 📌 Step 8: Encode Multi-label Stress Relief Activities
df['stress_relief_activities'] = df['stress_relief_activities'].fillna('')
activities = df['stress_relief_activities'].str.get_dummies(sep=', ')
df = pd.concat([df, activities], axis=1)
df.drop('stress_relief_activities', axis=1, inplace=True)

# 📌 Step 9: Ensure only numeric columns are used for normalization
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'depression' in numerical_cols:
    numerical_cols.remove('depression')  # exclude target
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 📌 Step 10: Correlation (now safe — only numeric columns remain)
correlation = df.corr()['depression'].sort_values(ascending=False)
selected_features = correlation[1:11].index.tolist()
final_df = df[selected_features + ['depression']]


# ✅ Confirm
print("\n✅ Final dataset exported.")
print("Selected Features:", selected_features)
print("Shape:", final_df.shape)
print(final_df.head())

# ---------------- VISUALIZATIONS ----------------

# Set global style
sns.set(style="whitegrid")

# Plot 1: Depression Class Count
plt.figure(figsize=(6, 4))
sns.countplot(x='depression', data=df)
plt.title("🧠 Depression Class Distribution")
plt.xlabel("Depression (0 = No, 1 = Yes)")
plt.ylabel("Number of Students")
plt.show()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title("🔗 Correlation Heatmap")
plt.show()

# Plot 3: CGPA vs Depression
plt.figure(figsize=(6, 4))
sns.boxplot(x='depression', y='cgpa', data=df)
plt.title("📚 CGPA by Depression Level")
plt.xlabel("Depression")
plt.ylabel("Normalized CGPA")
plt.show()

# Plot 4: Average Sleep vs Depression
plt.figure(figsize=(6, 4))
sns.violinplot(x='depression', y='average_sleep', data=df)
plt.title("😴 Average Sleep by Depression Level")
plt.xlabel("Depression")
plt.ylabel("Normalized Sleep Duration")
plt.show()

# Plot 5: Stress Relief Activity Usage (Top 5)
activity_counts = activities.sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 4))
sns.barplot(x=activity_counts.values, y=activity_counts.index, palette="viridis")
plt.title("🎯 Top 5 Stress Relief Activities")
plt.xlabel("Number of Students")
plt.ylabel("Activity")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = final_df.drop('depression', axis=1)
y = final_df['depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

from sklearn.tree import DecisionTreeClassifier, plot_tree

tree_model = DecisionTreeClassifier(max_depth=4)
tree_model.fit(X_train, y_train)

plt.figure(figsize=(15, 8))
plot_tree(tree_model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree for Depression Prediction")
plt.show()

