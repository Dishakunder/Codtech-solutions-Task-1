import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # PIL is required for handling images
import numpy as np
import pandas as pd
import joblib

# Load the trained SVM model and scaler
svm_model = joblib.load('svm_bmi_model_tuned.pkl')
scaler = joblib.load('svm_scaler_tuned.pkl')
le_gender = joblib.load('svm_le_gender_tuned.pkl')
le_category = joblib.load('svm_le_category_tuned.pkl')

def calculate_bmi_svm():
    try:
        # Fetching inputs from the user
        height = float(entry_height.get())
        weight = float(entry_weight.get())
        age = float(entry_age.get())
        gender = var_gender.get()
        
        # Validate gender input
        if gender not in ['Female', 'Male']:
            raise ValueError("Invalid gender selected")
        
        # Calculate BMI
        bmi = weight / (height / 100) ** 2
        
        # Prepare input for model prediction
        gender_encoded = le_gender.transform([gender])[0]  # Extract the scalar value
        X = pd.DataFrame([[height, weight, age, gender_encoded]], 
                         columns=['Height', 'Weight', 'Age', 'Gender'])
        
        # Scale the input
        X_scaled = scaler.transform(X)
        
        # Predict BMI category using the SVM model
        prediction = svm_model.predict(X_scaled)
        
        # Decode the prediction
        bmi_category_ml = le_category.inverse_transform(prediction)
        
        # Display the result
        messagebox.showinfo("BMI Result", 
                            f"Your BMI: {bmi:.2f}\n" +
                            f"ML Model Category: {bmi_category_ml[0]}\n")
    except ValueError as e:
        messagebox.showerror("Input Error", "Please enter valid values.")
    except Exception as e:
        messagebox.showerror("Error", "An unexpected error occurred.")

# Create the main application window
root = tk.Tk()
root.title("BMI Calculator - SVM")
root.geometry("500x400")
root.config(bg="#add8e6")  # Light blue background
root.resizable(False, False)

# Add a logo image
logo_image = Image.open("C:/Users/Lenovo/OneDrive/画像/bmi.png")  # Load your image
logo_image = logo_image.resize((100, 100), Image.Resampling.LANCZOS)
logo = ImageTk.PhotoImage(logo_image)

# Create a label for the logo
logo_label = tk.Label(root, image=logo, bg="#add8e6")
logo_label.grid(row=0, columnspan=5, pady=(10, 20))

# Heading
heading = tk.Label(root, text="BMI Calculator", font=("Helvetica", 24, "bold"), bg="#add8e6")
heading.grid(row=1, columnspan=10, pady=(0, 20))

# Height input
tk.Label(root, text="Height (cm):", font=("Helvetica", 12), bg="#add8e6").grid(row=2, column=0, padx=10, pady=5)
entry_height = tk.Entry(root, font=("Helvetica", 12))
entry_height.grid(row=2, column=1, pady=5)

# Weight input
tk.Label(root, text="Weight (kg):", font=("Helvetica", 12), bg="#add8e6").grid(row=3, column=0, padx=10, pady=5)
entry_weight = tk.Entry(root, font=("Helvetica", 12))
entry_weight.grid(row=3, column=1, pady=5)

# Age input
tk.Label(root, text="Age:", font=("Helvetica", 12), bg="#add8e6").grid(row=4, column=0, padx=10, pady=5)
entry_age = tk.Entry(root, font=("Helvetica", 12))
entry_age.grid(row=4, column=1, pady=5)

# Gender input
tk.Label(root, text="Gender:", font=("Helvetica", 12), bg="#add8e6").grid(row=5, column=0, padx=10, pady=5)
var_gender = tk.StringVar()
tk.Radiobutton(root, text="Female", variable=var_gender, value='Female', font=("Helvetica", 12), bg="#add8e6").grid(row=5, column=1, sticky="w")
tk.Radiobutton(root, text="Male", variable=var_gender, value='Male', font=("Helvetica", 12), bg="#add8e6").grid(row=5, column=2, sticky="w")

# Calculate button
calculate_button = tk.Button(root, text="Calculate BMI", font=("Helvetica", 14), command=calculate_bmi_svm, bg="#4CAF50", fg="white")
calculate_button.grid(row=6, columnspan=3, pady=20)

# Start the GUI event loop
root.mainloop()
