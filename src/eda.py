import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    print("\nClass distribution:\n", df["Class"].value_counts())

    #Class Distribution
    sns.countplot(x="Class", data=df)
    plt.title("Class Distribution (0=Normal, 1=Fraud)")
    plt.show()
    
    #Amount Distribution
    plt.figure()
    sns.histplot(df, x="Amount", hue="Class", bins=50, log_scale=True)
    plt.title("Transaction Amount Distribution by Class")
    plt.show()

    return df
run_eda("data/creditcard.csv")