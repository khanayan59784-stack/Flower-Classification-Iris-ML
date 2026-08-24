# 🌸 Flower Classification using Machine Learning

A Machine Learning project that classifies Iris flowers into **Setosa, Versicolor, and Virginica** using flower measurements.

This project performs **Exploratory Data Analysis (EDA), data visualization, classification, model comparison, confusion matrix analysis, model saving, and CLI-based flower prediction**.

---

## 🎯 Project Objective

The main objective of this project is to build a Machine Learning system that can predict the species of an Iris flower based on four measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The project uses two classification algorithms:

* Logistic Regression
* Decision Tree

---

## 📊 Dataset

This project uses the built-in **Iris dataset** provided by Scikit-learn.

The dataset contains **150 samples** belonging to three flower species:

| Species         | Description |
| --------------- | ----------- |
| Iris Setosa     | Class 0     |
| Iris Versicolor | Class 1     |
| Iris Virginica  | Class 2     |

### Features

| Feature      | Description         |
| ------------ | ------------------- |
| Sepal Length | Length of the sepal |
| Sepal Width  | Width of the sepal  |
| Petal Length | Length of the petal |
| Petal Width  | Width of the petal  |

---

## 🔍 Exploratory Data Analysis (EDA)

The project performs several EDA steps to understand the dataset:

* Dataset shape
* First five rows
* Missing-value checking
* Species distribution
* Statistical summary
* Feature correlation

---

## 📈 Data Visualizations

The project generates the following visualizations:

### 1. Species Distribution

Shows the number of samples belonging to each Iris species.

### 2. Feature Pair Plot

Visualizes relationships between different flower measurements and helps identify patterns between species.

### 3. Correlation Heatmap

Shows the correlation between the four numerical features.

### 4. Model Accuracy Comparison

Compares the accuracy of Logistic Regression and Decision Tree.

### 5. Confusion Matrix

Shows correct and incorrect predictions made by the Logistic Regression model.

---

## 🤖 Machine Learning Models

### Logistic Regression

Logistic Regression is used as a classification algorithm to predict the species of an Iris flower based on its four measurements.

### Decision Tree

Decision Tree classifies flowers by creating a sequence of decision rules based on the input features.

Both models are trained and evaluated using the same training and testing datasets.

---

## 🧪 Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

Stratified splitting is used to maintain the class distribution between training and testing data.

---

## 📊 Model Evaluation

The models are evaluated using **Accuracy Score**.

The project compares:

* Logistic Regression Accuracy
* Decision Tree Accuracy

The model with the higher accuracy is automatically selected as the **Best Model**.

---

## 🔲 Confusion Matrix

A confusion matrix is generated for the Logistic Regression model.

It shows how many flowers were:

* Correctly classified
* Incorrectly classified

It also helps identify possible misclassifications between different Iris species.

---

## 💾 Model Saving

After comparing both models, the better-performing model is automatically saved using Joblib.

The saved model is stored at:

```text
models/iris_model.pkl
```

This saved model is then used by the separate prediction script.

---

## 🌺 CLI Flower Prediction

The project includes a separate `predict.py` script for predicting a new flower.

The user enters:

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

The saved Machine Learning model then predicts the flower species.

### Example

```text
Enter Sepal Length: 5.1
Enter Sepal Width: 3.5
Enter Petal Length: 1.4
Enter Petal Width: 0.2

Predicted Flower: setosa
```

---

## 📁 Project Structure

```text
Flower-Classification-Iris-ML/
│
├── flower_classification.py
├── predict.py
├── requirements.txt
├── README.md
│
├── models/
│   └── iris_model.pkl
│
├── species_distribution.png
├── feature_pairplot.png
├── correlation_heatmap.png
├── accuracy_comparison.png
└── confusion_matrix.png
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Flower-Classification-Iris-ML.git
```

Open the project folder:

```bash
cd Flower-Classification-Iris-ML
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run the complete Machine Learning project

```bash
python flower_classification.py
```

This will:

1. Load the Iris dataset
2. Perform EDA
3. Generate visualizations
4. Train both models
5. Compare their accuracy
6. Generate the confusion matrix
7. Select the best model
8. Save the model
9. Perform a sample prediction

### Run the CLI prediction

After the model has been created, run:

```bash
python predict.py
```

Enter the four flower measurements when prompted.

---

## 📌 Expected Output

The program displays:

* Dataset information
* Missing values
* Species count
* Statistical summary
* Training and testing data size
* Logistic Regression accuracy
* Decision Tree accuracy
* Best model
* Sample prediction
* Model saved confirmation

---

## 🚀 Future Improvements

Future versions of this project could include:

* Web-based prediction interface
* Streamlit application
* Interactive dashboards
* Additional Machine Learning algorithms
* Hyperparameter tuning
* Model performance reports
* Cloud deployment

---

## 👨‍💻 Author

**Pathan Ayan Asif**

---

## ⭐ Project Highlights

This project demonstrates the complete Machine Learning workflow:

**Data Loading → EDA → Visualization → Train/Test Split → Model Training → Model Comparison → Evaluation → Model Saving → New Prediction**

---

⭐ If you find this project useful, consider giving the repository a star.
