from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess(df):
    df = df.drop(columns=["Time"])

    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res