THIS README IS MADE USING THE HELP OF CHATGPT

# 🩺 Diabetes Prediction Using Machine Learning

This project is a beginner-friendly implementation of a machine learning model that predicts whether a person is likely to have diabetes, based on medical data. The dataset used is the Pima Indians Diabetes Dataset which you can get by clicking the below link

Diabetes Dataset= https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data


---

## 📌 Project Overview

This project performs the following tasks:

- Loads and visualizes the diabetes dataset
- Cleans the data (replaces invalid 0s with median values)
- Splits the data into training and test sets
- Scales the features using StandardScaler
- Trains a Logistic Regression model
- Evaluates the model using accuracy score

---

## 📊 Dataset Information

The dataset contains several medical predictor variables and one target variable ("Outcome"):

| Feature          | Description                      |
|------------------|----------------------------------|
| Pregnancies      | Number of times pregnant         |
| Glucose          | Plasma glucose concentration     |
| BloodPressure    | Diastolic blood pressure         |
| SkinThickness    | Triceps skin fold thickness      |
| Insulin          | 2-Hour serum insulin             |
| BMI              | Body Mass Index                  |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age              | Age (years)                      |
| Outcome          | 1 = diabetic, 0 = non-diabetic   |

**Note:** Some features have 0s which are invalid (e.g., Glucose = 0). These are cleaned during preprocessing.

---

## 🛠️ Installation & Setup

1. **Clone the repository** or download the project folder.
2. **Ensure you have Python installed (3.1 or higher).**
3. Open the project in **Visual stuido code** or any code editor.
4. Open a notebook file
### 📦 Required Libraries


**Importing required libraries**
## Install if its not available in your pc
```python
!pip install pandas matplotlib seaborn scikit-learn
```
## Import necessary libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```

## Load the dataset
--Download the dataset and replace the path. Below is the link to download
Diabetes Dataset= https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data

```python
data = pd.read_csv("C:\\Users\\robin\\OneDrive\\Desktop\\Rocheston Project\\diabetes.csv")
#or can use data = pd.read_csv(r"C:\Users\robin\OneDrive\Desktop\Rocheston Project\diabetes.csv")
data
```

## Cleaning Data
```python
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    data[col] = data[col].replace(0, data[col].median())
```

## Checking for missing data(NULL)
```python
sns.heatmap(data.isnull())
![alt text](image.png)
```
## Correlation Matrix of the data
```python
correlation = data.corr()
print( correlation)
```
## Visualizing the Heat Map
```python
sns.heatmap(correlation)
```
## Train Test Split 
```python
x=data.drop("Outcome", axis=1)
y=data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train
```
## Scale Features
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
```
## Training the model
```python
model=LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
```
## Making Predictions
```python
predictions = model.predict(x_test)
```
## Evaluation
```python
accuracy = accuracy_score(y_test, predictions)

print(accuracy)
```

# 🖥️ Building the GUI with Django

This section explains how to build a simple web interface for the Diabetes Prediction model using Django.

---

## 📁 Step 1: Set Up the Project Directory

1. Create a folder named `Project` on your Desktop.
2. Save your Jupyter Notebook or Python script (from previous steps) into this folder.
3. Open the `Project` folder in **Visual Studio Code**.

---

## 🛠️ Step 2: Create a Django Project

Open the **terminal** in Visual Studio Code and run the following commands:

```bash
# Create a new Django project
django-admin startproject DiabetesPrediction

# Navigate into the project folder
cd DiabetesPrediction

# Start the Django development server
python manage.py runserver
```
## 🌐 Step 4: Create and Edit the Web Page

Now we'll create the homepage for the web app.

---

### 📂 Create the `templates` Folder

1. Navigate to the `DiabetesPrediction` folder (where `manage.py` is located).
2. Create a new folder named **`templates`**.
3. Inside the `templates` folder, create a new HTML file named **`home.html`**.

📌 Ensure that the `templates` folder is **in the same directory as `manage.py`**, not inside a subfolder.

---

### 📝 Add the Following Code to `home.html`
```html
{% load static %}

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Home</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
            background-image: url("{% static 'DiabetesPrediction/images/img2.jpg' %}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #fff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }

        .container {
            text-align: center;
            background-color: rgba(255, 255, 255, 0.1);
            padding: 60px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
        }

        h1 {
            font-size: 64px;
            margin-bottom: 20px;
            color: #fff;
        }

        input[type="submit"] {
            background-color: #007BFF;
            color: #fff;
            padding: 15px 40px;
            border: none;
            border-radius: 10px;
            font-size: 20px;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }

        input[type="submit"]:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welcome to Diabetes Prediction</h1>
        <form action="predict">
            <input type="submit" value="Let's Get Started">
        </form>       
    </div>
</body>
</html>
```


## Next open up the urls.py in the DiabetesPrediction folder and paste the below code
```python
"""
URL configuration for DiabetesPrediction project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("",views.home),
    path("predict/",views.predict),
    path("predict/result",views.result),
    
]
```

## Next create views.py in  the DiabetesPrediction folder 
**The views.py should be in the same level as settings.py , urls.py and etc**

# Add the below code in views.py
```python
from django.shortcuts import render

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from django.shortcuts import render

def home(request):
    return render(request,'home.html')

def predict(request):
    return render(request,'predict.html')

from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def result(request):
    # Load dataset
    data = pd.read_csv(r"C:\Users\robin\OneDrive\Desktop\Rocheston Project\diabetes.csv")

    # Replace 0s with median in selected features
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        data[col] = data[col].replace(0, data[col].median())

    # Split X and y
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Get values from form (default to '0')
    input_values = {}
    val = []
    for i in range(1, 9):
        key = f'n{i}'
        value = request.GET.get(key, '0')
        input_values[key] = value
        val.append(float(value))

    # Replace 0s with median for specific features (same as training)
    for i, col in enumerate(X.columns):
        if val[i] == 0 and col in cols_to_fix:
            val[i] = data[col].median()

    # Scale input
    val_scaled = scaler.transform([val])

    # Predict
    prediction = model.predict(val_scaled)
    result2 = "Diabetes Positive" if prediction[0] == 1 else "Diabetes Negative"

    # Send prediction and input values back to template
    context = input_values
    context['result2'] = result2
    return render(request, 'predict.html', context)
```

## Next open the settings.py in the DiabetesPrediction folder and change the 'DIRS':[] in the templates section
**Import os if its not available in the settings.py if there is errors regarding os**
```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        #Update the below code add os.path.join(BASE_DIR, 'templates') in the []
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

## 🖼️ How to Add a Background Image to the Web Page

To set a background image on your Django web page, follow these steps:

---

### 📁 Step 1: Create the `static` Folder

1. Inside the `DiabetesPrediction` folder (where `settings.py` and `views.py` are located), create a new folder named **`static`**.
2. Inside the `static` folder, create another folder named **`DiabetesPrediction`** (this is your app-specific static directory).
3. Inside the `DiabetesPrediction` folder, create one more folder named **`images`**.
4. Place the image you want to use as a background inside this `images` folder.

> ✅ You can simply drag and drop the image into the `images` folder.

---

### 🧾 Step 2: Reference the Image in `home.html`

In your `home.html` file, update the `background-image` line to use the **Django static path** by copyng the relative path of the image and pasting into it make sure remove unnecessary path **It should be similar like below**:

```html
background-image: url("{% static 'DiabetesPrediction/images/img2.jpg' %}");
```
---

## 🧾 Step 5: Create the Prediction Page (`predict.html`)

Next, let’s build the prediction form that allows users to enter patient details and get diabetes prediction results.

---

### 📂 Create `predict.html` Inside the `templates` Folder

1. Go to your `templates` folder (created earlier in the same directory as `manage.py`).
2. Create a new file named **`predict.html`**.
3. Paste the following HTML code into the `predict.html` file.

---

### 🧑‍⚕️ `predict.html` Code

```html
{% load static %}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Diabetes Prediction</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-image: url("{% static 'DiabetesPrediction/images/img2.jpg' %}");
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-size: cover;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }

        .main {
            background-color: rgba(124, 69, 64, 0.95);
            border-radius: 15px;
            padding: 30px 40px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            max-width: 600px;
            width: 100%;
        }

        h1 {
            color: #000000;
            font-size: 32px;
            text-align: center;
            margin-bottom: 30px;
        }

        table {
            width: 100%;
        }

        td {
            padding: 10px;
            font-size: 16px;
        }

        input[type="range"] {
            width: 100%;
        }

        .value-label {
            margin-left: 10px;
            font-weight: bold;
            color: #000000;
        }

        input[type="submit"] {
            background-color: #4CAF50;
            color: rgb(0, 0, 0);
            padding: 10px 20px;
            border: none;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            display: block;
            margin: 20px auto 0;
            transition: background-color 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #45a049;
        }

        .result {
            margin-top: 20px;
            font-weight: bold;
            text-align: center;
            color: #d9534f;
            font-size: 18px;
        }
    </style>
</head>
<body>
<div class="main">
    <h1>Enter Patient Information</h1>
    <form action="result" method="get">
        <table>
            <tr>
                <td align="right">Pregnancies:</td>
                <td>
                    <input type="range" name="n1" id="n1" min="0" max="17" step="1" value="{{ n1|default:'0' }}" oninput="updateValue(this)">
                    <span class="value-label" id="n1_val">{{ n1|default:'0' }}</span>
                </td>
            </tr>
            <tr>
                <td align="right">Glucose:</td>
                <td>
                    <input type="range" name="n2" id="n2" min="0" max="200" step="1" value="{{ n2|default:'0' }}" oninput="updateValue(this)">
                    <span class="value-label" id="n2_val">{{ n2|default:'0' }}</span>
                </td>
            </tr>
            <tr>
                <td align="right">Blood Pressure:</td>
                <td>
                    <input type="range" name="n3" id="n3" min="0" max="122" step="1" value="{{ n3|default:'0' }}" oninput="updateValue(this)">
                    <span class="value-label" id="n3_val">{{ n3|default:'0' }}</span>
                </td>
            </tr>
            <tr>
                <td align="right">Skin Thickness:</td>
                <td>
                    <input type="range" name="n4" id="n4" min="0" max="99" step="1" value="{{ n4|default:'0' }}" oninput="updateValue(this)">
                    <span class="value-label" id="n4_val">{{ n4|default:'0' }}</span>
                </td>
            </tr>
            <tr>
                <td align="right">Insulin:</td>
                <td>
                    <input type="range" name="n5" id="n5" min="0" max="846" step="1" value="{{ n5|default:'0' }}" oninput="updateValue(this)">
                    <span class="value-label" id="n5_val">{{ n5|default:'0' }}</span>
                </td>
            </tr>
            <tr>
                <td align="right">BMI:</td>
                <td>
                    <input type="range" name="n6" id="n6" min="10" max="67" step="0.1" value="{{ n6|default:'0' }}" oninput="updateValue(this)">
                    <span class="value-label" id="n6_val">{{ n6|default:'0' }}</span>
                </td>
            </tr>
            <tr>
                <td align="right">Diabetes Pedigree Function:</td>
                <td>
                    <input type="range" name="n7" id="n7" min="0.1" max="2.5" step="0.01" value="{{ n7|default:'0.1' }}" oninput="updateValue(this)">
                    <span class="value-label" id="n7_val">{{ n7|default:'0.1' }}</span>
                </td>
            </tr>
            <tr>
                <td align="right">Age:</td>
                <td>
                    <input type="range" name="n8" id="n8" min="0" max="100" step="1" value="{{ n8|default:'0' }}" oninput="updateValue(this)">
                    <span class="value-label" id="n8_val">{{ n8|default:'0' }}</span>
                </td>
            </tr>
        </table>
        <input type="submit" value="Predict">
    </form>
    <div class="result">Result: {{ result2 }}</div>
</div>

<script>
    function updateValue(slider) {
        const span = document.getElementById(slider.id + "_val");
        span.textContent = slider.value;
    }
</script>
</body>
</html>

```
> ✅ **Reminder:**  
> Make sure your image is stored in the following path:  
> `static/DiabetesPrediction/images/img2.jpg`  
>
> ✅ **To reference the image correctly in Django:**  
> ```html
> background-image: url("{% static 'DiabetesPrediction/images/img2.jpg' %}");
> ```

## ✅ Final Step: Run Your Project

Make sure all files are saved, and then rerun your Django server:

```bash
python manage.py runserver
```
---

# 🎉 Congratulations!  
You've successfully built a machine learning web app with a functional GUI using **Django** and **Logistic Regression**.

✅ Feel free to:
- Improve the UI for a better user experience  
- Try out different machine learning models (e.g., Decision Tree, Random Forest, etc.)  
- Deploy your project for real-world usage 🚀

Great job and keep coding! 💻✨





 
