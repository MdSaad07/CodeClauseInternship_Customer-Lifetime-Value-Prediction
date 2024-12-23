# Customer-Lifetime-Value-Prediction  
💼 **Customer Lifetime Value Prediction Using Machine Learning**

### 🎯 **Objective:**  
The goal of this project is to build a machine learning model that predicts **Customer Lifetime Value (CLV)** based on features such as purchase history, customer demographics, engagement metrics, and interaction frequency. The model will allow businesses to input customer details and receive an estimated lifetime value in real-time. This tool aims to assist organizations in making informed decisions about customer segmentation, retention strategies, and targeted marketing efforts.

---

### 📊 **Model Accuracy:**  
- **MAE (Mean Absolute Error):** 373.974  
  - This indicates that, on average, the model's prediction for CLV deviates by about 373.974 units from the actual value. Given the maximum CLV value is 16,624.75, this error margin is relatively low.  
- **MAPE (Mean Absolute Percentage Error):** 4.8%  
  - This means that the model's predictions deviate by around 4.8% from the actual CLV values on average, which is a good indication of model accuracy.  

---

### 🛠️ **Tech Stack:**  
**Development Environment:**  
- 🖥️ **VS Code:** For writing and executing code.  
- 📓 **Jupyter Notebook:** Used within VS Code for running Python scripts interactively.  

**Programming Language:**  
- 🐍 **Python:** The entire project is implemented in Python.  

**Machine Learning Libraries:**  
- 📚 **Scikit-learn:** For training machine learning regression models to predict CLV.  
- 🐼 **Pandas** & 🧮 **NumPy:** For data preprocessing and manipulation.  
- 📊 **Matplotlib:** For data visualization.  

---

### 🛤️ **Roadmap:**  

#### 1️⃣ **Understanding the Problem and Dataset:**  
- **Objective:** Predict CLV based on several features.  
- **Dataset:** Contains the following features:  
  - Vehicle Class, Coverage, Renew Offer Type, Employment Status, Marital Status, Education, Number of Policies, Monthly Premium Auto, Total Claim Amount, Income, Customer Lifetime Value.  

#### 2️⃣ **Data Collection & Exploration:**  
- 📊 **Acquire the Dataset:** The dataset includes customer and policy-related information, available from relevant data sources.  
- 🔍 **Exploratory Data Analysis (EDA):**  
  - **Unique Values:** Checked for each feature's unique values to understand categorical data.  
  - **Normal Distribution:** Verified if the data follows a normal distribution for continuous variables.  
  - **Missing Values:** Checked for any missing values within the dataset and handled them appropriately.  
  - **Outliers:** Used the **Interquartile Range (IQR)** method to detect outliers, especially in **Total Claim Amount** and **Customer Lifetime Value**, which were removed to improve model accuracy.  

#### 3️⃣ **Data Preprocessing:**  
- 🧹 **Handling Missing Values:** Missing values were imputed or dropped depending on the feature.  
- 🏗️ **Feature Engineering:** 
  - Categorical variables (like Employment Status and Marital Status) were encoded into numerical format (one-hot encoding).  
  - Scaled numerical features like **Income** and **Monthly Premium Auto** if needed.  
- ✂️ **Train-Test Split:** Split the dataset into training and testing sets with a **70:30** ratio using `train_test_split` from Scikit-learn.  

#### 4️⃣ **Model Building & Model Selection:**  
- **Algorithms Used:**  
  - Linear Regression  
  - KNeighborsRegressor  
  - DecisionTreeRegressor  
  - RandomForestRegressor  
  - AdaBoostRegressor  
  - XGBRegressor  
  - GradientBoostingRegressor  

  **Note:**  
  - **Gradient Boosting** and **Random Forest** were primarily used because they showed the **lowest RMSE** among all the models.  
  - **Model Testing:**  
    - Trained and evaluated each model using the training and test datasets.  

#### 5️⃣ **Testing the Output:**  
- **Model's Performance:** Evaluated using MAE and MAPE.  
  - MAE of 373.974 and MAPE of 4.8% show a reasonable prediction accuracy.  

---

### 👨‍💻 **Author:**  
**Mohammed Saad Fazal**
