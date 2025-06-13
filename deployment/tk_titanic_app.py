import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load the model
model = joblib.load(r"random_forest_model.pkl")

class TitanicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Titanic Survival Predictor")
        self.root.geometry("400x600")

        # Input fields
        self.entries = {}
        fields = [
                'pclass', 'sibsp', 'parch', 'alone', 'age_norm', 'fare_norm',
                'sex_encoded', 'embarked_Q', 'embarked_S', 'class_Second', 'class_Third',
                ]
        for field in fields:
            label = tk.Label(root, text=field.upper())
            label.pack()
            entry = tk.Entry(root)
            entry.pack()
            self.entries[field] = entry

        # Predict button
        tk.Button(root, text="Predict", command=self.predict).pack(pady=10)

        # Result label
        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack()

    def predict(self):
        try:
            pclass = float(self.entries['pclass'].get())
            sibsp = float(self.entries['sibsp'].get())
            parch = float(self.entries['parch'].get())
            alone = float(self.entries['alone'].get())
            age_norm = float(self.entries['age_norm'].get())
            fare_norm = float(self.entries['fare_norm'].get())
            sex_encoded = float(self.entries['sex_encoded'].get())
            embarked_Q = float(self.entries['embarked_Q'].get())
            embarked_S = float(self.entries['embarked_S'].get())
            class_Second = float(self.entries['class_Second'].get())
            class_Third = float(self.entries['class_Third'].get())

            X = np.array([[pclass, sibsp, parch, alone, age_norm, fare_norm,
                        sex_encoded, embarked_Q, embarked_S, class_Second, class_Third]])

            prediction = model.predict(X)[0]

            self.result_label.config(
                text=f"Prediction: {'Survived' if prediction == 1 else 'Did NOT Survive'}"
            )
        except Exception as e:
            self.result_label.config(text=f"Error: {e}")
if __name__ == "__main__":
    root = tk.Tk()
    app = TitanicApp(root)
    root.mainloop()
