from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/creditcard.csv")

def preprocess(df):
    df = df.drop(columns=["Time"])
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # print("X_train shape:", X_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test

X, y, scaler = preprocess(df)
X_train, X_test, y_train, y_test = split_data(X, y)

#Data Balance
def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    plt.figure(figsize=(6,4))
    y_res.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Class Distribution After SMOTE")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()

    return X_res, y_res

X_train_res, y_train_res = balance_data(X_train, y_train)


    