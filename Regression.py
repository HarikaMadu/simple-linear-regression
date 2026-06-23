import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 30, 35, 50, 55, 65, 70, 85]
}

df = pd.DataFrame(data)

# Independent and dependent variables
X = df[["Study_Hours"]]
y = df["Marks"]

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
hours = int(input("Enter study hours: "))
predicted_marks = model.predict([[hours]])

print(f"Predicted Marks: {predicted_marks[0]:.2f}")

# Graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Simple Linear Regression")
plt.show()