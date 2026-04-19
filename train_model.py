import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Heart.csv")

print("Dataset shape before cleaning:", df.shape)
print("\nMissing values before cleaning:\n")
print(df.isnull().sum())

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

if 'ChestPain' in df.columns and df['ChestPain'].dtype == 'object':
    df['ChestPain'] = df['ChestPain'].astype("category").cat.codes

if 'Thal' in df.columns and df['Thal'].dtype == 'object':
    df['Thal'] = df['Thal'].astype("category").cat.codes

if 'AHD' in df.columns and df['AHD'].dtype == 'object':
    df['AHD'] = df['AHD'].astype("category").cat.codes

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values after cleaning:\n")
print(df.isnull().sum())

X = df.drop("AHD", axis=1)
y = df["AHD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel trained and saved successfully as model.pkl")