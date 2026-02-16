from src.eda import run_eda
from src.preprocess import preprocess, split_data, balance_data
from src.training import train_model
from src.evaluation import evaluate_model
import joblib

#EDA
df = run_eda("data/creditcard.csv")

#Preprocess
X, y = preprocess(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_res, y_train_res = balance_data(X_train, y_train)

#Model
model = train_model(X_train_res, y_train_res)

#Evaluation
# evaluate_model(model, X_test, y_test)

joblib.dump(model, "models/model.pkl")
print("Model saved successfully!")

