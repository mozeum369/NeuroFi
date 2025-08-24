import os

# Define the path to the bot.py file
file_path = "src/bot.py"

# Define the train_model function to be added
train_model_function = '''
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_model(log_data):
    X = log_data.drop("label", axis=1)
    y = log_data["label"]

    if len(np.unique(y)) < 2:
        print("Not enough class diversity to train model. Skipping training.")
        return None

    model = LogisticRegression()
    model.fit(X, y)
    return model
'''

# Create the file and write the function if it doesn't exist
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, "w") as f:
    f.write(train_model_function.strip())

print("train_model function has been added to src/bot.py.")

