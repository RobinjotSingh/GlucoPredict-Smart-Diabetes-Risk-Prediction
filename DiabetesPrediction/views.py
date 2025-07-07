from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Load and clean dataset
    data = pd.read_csv(r"C:\Users\robin\OneDrive\Desktop\Rocheston Project\diabetes.csv")
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        data[col] = data[col].replace(0, data[col].median())

    # Prepare data
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Get and process input
    input_values = {}
    val = []
    for i, col in enumerate(X.columns):
        key = f'n{i+1}'
        value = request.GET.get(key, '0')
        float_val = float(value)
        if float_val == 0 and col in cols_to_fix:
            float_val = data[col].median()
        input_values[key] = value
        val.append(float_val)

    val_scaled = scaler.transform([val])
    prediction = model.predict(val_scaled)
    input_values['result2'] = "Diabetes Positive" if prediction[0] == 1 else "Diabetes Negative"

    return render(request, 'predict.html', input_values)
