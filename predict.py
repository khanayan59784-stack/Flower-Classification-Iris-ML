# ==========================================
# IRIS FLOWER CLI PREDICTION
# ==========================================

import joblib
from sklearn.datasets import load_iris


# Load Iris information
iris = load_iris()

# Load saved model
model = joblib.load(
    "models/iris_model.pkl"
)


print("\n================================")
print("     IRIS FLOWER PREDICTION")
print("================================")


# User inputs

sepal_length = float(
    input("Enter Sepal Length: ")
)

sepal_width = float(
    input("Enter Sepal Width: ")
)

petal_length = float(
    input("Enter Petal Length: ")
)

petal_width = float(
    input("Enter Petal Width: ")
)


# Create input data

flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]


# Prediction

prediction = model.predict(flower)


# Convert number to flower name

flower_name = iris.target_names[
    prediction[0]
]


print("\n================================")
print("          RESULT")
print("================================")

print("Predicted Flower:", flower_name)