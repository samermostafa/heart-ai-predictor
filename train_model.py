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

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())


chestpain_mapping = {
    "typical": 0,
    "nontypical": 1,
    "nonanginal": 2,
    "asymptomatic": 3
}

thal_mapping = {
    "normal": 0,
    "fixed": 1,
    "reversable": 2
}

ahd_mapping = {
    "No": 0,
    "Yes": 1
}

df["ChestPain"] = df["ChestPain"].map(chestpain_mapping)
df["Thal"] = df["Thal"].map(thal_mapping)
df["AHD"] = df["AHD"].map(ahd_mapping)


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
