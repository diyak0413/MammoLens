from tkinter import *
from tkinter import messagebox, filedialog
from tkinter.ttk import Style
from models.model import evaluate_model, load_model
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
import os
from pathlib import Path
from PIL import Image, ImageTk


# Define color scheme
BACKGROUND_COLOR = "#FFFFFF"
PRIMARY_COLOR = "#007BFF"
SECONDARY_COLOR = "#6C757D"
TEXT_COLOR = "#000000"


class Header(Frame):
    def __init__(self, master, title):
        super().__init__(master, bg=PRIMARY_COLOR)
        self.title = Label(self, text=title, font=("Helvetica", 16, "bold"), bg=PRIMARY_COLOR, fg=BACKGROUND_COLOR)
        self.title.pack(pady=10)



class FirstPage(Frame):
    """
    The FirstPage class represents the initial page of the application,
    where the user inputs patient data and selects how to input examination
    results - from a CSV file or manually.
    """
    def __init__(self, app):
        Frame.__init__(self, app.root)
        self.app = app
        self.configure(bg=BACKGROUND_COLOR)

        # Creating and configuring widgets
        self.label_info = Label(self, text="ENTER PATIENT DATA", fg=PRIMARY_COLOR, bg=BACKGROUND_COLOR, font=("Helvetica", 18, 'bold'))
        self.label_name = Label(self, text="First Name:", fg=TEXT_COLOR, bg=BACKGROUND_COLOR, font=("Helvetica", 12))
        self.entry_name = Entry(self, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Helvetica", 12))
        self.label_surname = Label(self, text="Last Name:", fg=TEXT_COLOR, bg=BACKGROUND_COLOR, font=("Helvetica", 12))
        self.entry_surname = Entry(self, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Helvetica", 12))
        self.label_date = Label(self, text="Examination date:", fg=TEXT_COLOR, bg=BACKGROUND_COLOR, font=("Helvetica", 12))
        self.entry_date = Entry(self, bg=BACKGROUND_COLOR, fg=SECONDARY_COLOR, insertbackground=TEXT_COLOR, font=("Helvetica", 12))
        self.entry_date.insert(0, "dd-mm-yyyy")
        self.label_choice = Label(self, text="Patient's results:", fg=TEXT_COLOR, bg=BACKGROUND_COLOR, font=("Helvetica", 12))

        self.browse_button = Button(self, text="Load data from CSV file", command=self.process_data,
                                    bg=PRIMARY_COLOR, fg='black', font=("Helvetica", 12))
        self.manual_button = Button(self, text="Enter data manually", command=self.process_data2,
                                    bg=PRIMARY_COLOR, fg='black', font=("Helvetica", 12))
        self.help_button = Button(self, text="?", command=self.show_csv_help,
                                  bg=PRIMARY_COLOR, fg='black', font=("Helvetica", 12))

        # Layout
        self.label_info.pack(pady=(20, 10))
        self.label_name.pack(anchor=W, padx=20, pady=(10, 0))
        self.entry_name.pack(fill=X, padx=20, pady=(0, 10))
        self.label_surname.pack(anchor=W, padx=20, pady=(10, 0))
        self.entry_surname.pack(fill=X, padx=20, pady=(0, 10))
        self.label_date.pack(anchor=W, padx=20, pady=(10, 0))
        self.entry_date.pack(fill=X, padx=20, pady=(0, 10))
        self.label_choice.pack(anchor=W, padx=20, pady=(10, 0))
        self.browse_button.pack(fill=X, padx=20, pady=(10, 5))
        self.manual_button.pack(fill=X, padx=20, pady=5)
        self.help_button.pack(pady=10)

    def validate_date(self):
        """Validate the format and range of the entered date."""
        date = self.entry_date.get()
        if date == "dd-mm-yyyy":
            messagebox.showerror("Invalid Date", "Please enter the date.")
            return False

        if not re.match('\\d{2}-\\d{2}-\\d{4}', date):
            messagebox.showerror("Invalid Date", "Please enter a valid date in dd-mm-yyyy format.")
            return False

        day, month, year = map(int, date.split('-'))
        if day < 1 or day > 31 or month < 1 or month > 12 or year < 1950 or year > 2024:
            messagebox.showerror("Invalid Date", "Date is out of range. Please enter a valid date.")
            return False

        return True

    def validate_name_surname(self):
        """Validate the entered name and surname."""
        name = self.entry_name.get().strip()
        surname = self.entry_surname.get().strip()

        if not name or not surname:
            messagebox.showerror("Invalid Input", "Please enter both name and surname.")
            return False
        return True

    def validate_csv_data(self, file_path):
        """
        Validate the CSV data file.

        Args:
            file_path: The path to the CSV file to be validated.

        Checks for:
        - Whether the file can be read correctly.
        - If the file is not empty.
        - If all required columns are present.
        - If the values in each column are numeric, non-zero, and not missing.
        - If the values are within expected bounds.
        """
        # Attempt to read the CSV file
        try:
            csv_data = pd.read_csv(file_path)
        except Exception as e:
            messagebox.showerror("Error", "An error occurred while reading the CSV file. "
                                 "Please ensure the file format is correct and try again.")
            return False

        # Check if the CSV file is empty
        if csv_data.empty:
            messagebox.showerror("Invalid File", "The CSV file is empty.")
            return False

        # Iterate through each column defined in label_to_csv_column
        for csv_column, (label, category) in self.app.label_to_csv_column.items():
            # Check for missing columns in the CSV file
            if csv_column not in csv_data.columns:
                messagebox.showerror("Invalid File", f"Missing required column '{csv_column}' in CSV file.")
                return False

            # Validate each value in the column
            for value in csv_data[csv_column]:

                # Convert value to numeric and handle non-numeric values
                try:
                    numeric_value = pd.to_numeric(value)
                except ValueError:
                    messagebox.showerror("Invalid Data", f"Please enter a valid number for column '{csv_column}'.")
                    return False

                # Check for zero values
                if numeric_value == 0:
                    messagebox.showerror("Invalid Data", f"Value in column '{csv_column}' cannot be zero.")
                    return False

                # Check for missing (NaN) values
                if pd.isna(value):
                    messagebox.showerror("Invalid Data", f"Missing value in column '{csv_column}'.")
                    return False

                # Validate if the value is within the predefined bounds
                bounds = self.app.category_bounds[category][label]
                if not (bounds[0] <= numeric_value <= bounds[1]):
                    response = messagebox.askyesno(
                    "Value Out of Bounds",
                    f"The value in column '{csv_column}' may be out of typical bounds.\n"
                    "Do you want to continue anyway?"
                    )
                    if not response:  # If the user chooses 'No'
                        return False

        # Return True if all validations pass
        return True

    def clear_default_text(self, event):
        """Clear the default text when the date entry gains focus."""
        if self.entry_date.get() == "dd-mm-yyyy":
            self.entry_date.delete(0, END)
            self.entry_date.config(fg="lime green")  # Change font color to lime green when clicked on

    def restore_default_text(self, event):
        """Restores default text if the date entry is empty."""
        if not self.entry_date.get():
            self.entry_date.insert(0, "dd-mm-yyyy")
            self.entry_date.config(fg="grey")  # Change font color to grey if the field is empty

    def get_data(self):
        """Retrieve and return the user-entered patient data."""
        name = self.entry_name.get()
        surname = self.entry_surname.get()
        date = self.entry_date.get()

        return name, surname, date

    def browse_file(self):
        """Open file dialog to select CSV file and validate its content."""
        file_path = filedialog.askopenfilename(title="Choose CSV fILE", filetypes=[("CSV files", "*.csv")])
        if file_path:
            valid_data = self.validate_csv_data(file_path)
            if valid_data:
                self.predict_diagnosis(file_path)
            else:
                messagebox.showerror("Invalid File", "Please provide a valid CSV file with correct data format.")


    def predict_diagnosis(self,file_path):
        """Predict diagnosis using the loaded CSV data."""
        try:
            # Load data from CSV and predict diagnosis
            input_data = pd.read_csv(file_path)
            diagnosis_result = self.app.reg_model.predict(input_data)[0]
            self.app.show_result_page(diagnosis_result)

        except Exception as e:
            messagebox.showerror("Error", "An error occurred while processing the data. "
                                 "Please check the file format and data integrity, then try again.")

    def process_data(self):
        """Validate date and names before browsing CSV file."""
        if  self.validate_date() and self.validate_name_surname():
            self.browse_file()

    def process_data2(self):
        """Validate date and names before showing data page 1."""
        if self.validate_date() and self.validate_name_surname():
            self.app.show_data_page1()

    def reset_entries(self):
        """Clear all input fields."""
        self.entry_name.delete(0, END)
        self.entry_surname.delete(0, END)
        self.entry_date.delete(0, END)
        self.entry_date.insert(0, "dd-mm-yyyy")
        self.entry_date.config(fg="grey")

    def show_csv_help(self):
        """Show help message regarding the format of CSV file."""
        messagebox.showinfo("CSV File Format",
                            "The CSV file should contain the following columns:\n\n"
                            "radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, "
                            "compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean, "
                            "radius_se, texture_se, perimeter_se, area_se, smoothness_se, \n"
                            "compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, "
                            "radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, "
                            "compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst\n\n"
                            "Each column should contain numerical values. The file should have exactly one row of data.\n\n"
                            "For more information about the features, please refer to the features_description.pdf file.")


class ResultPage(Frame):
    """Class representing the result page of the user interface."""

    def __init__(self, app, first_page_reference):
        Frame.__init__(self, app.root)
        self.app = app
        self.first_page_reference = first_page_reference
        self.configure(bg=BACKGROUND_COLOR)

        self.label_result = Label(self, text="Result:", fg=TEXT_COLOR, bg=BACKGROUND_COLOR, font=("Helvetica", 16))
        self.label_result.pack(pady=20)

        self.result_label = Label(self, text="", fg=PRIMARY_COLOR, bg=BACKGROUND_COLOR, font=("Helvetica", 16, 'bold'))
        self.result_label.pack()

        self.button_back = Button(self, text="Back", command=app.show_previous_page,
                                  bg=PRIMARY_COLOR, fg="black", font=("Helvetica", 12))
        self.button_back.pack(pady=10)

        self.save_button = Button(self, text="Save data to a file", command=self.save_to_file,
                                  bg=PRIMARY_COLOR, fg="black", font=("Helvetica", 12))
        self.save_button.pack(pady=10)

        self.metrics_button = Button(self, text="Model info", command=self.show_metrics,
                                     bg=PRIMARY_COLOR, fg="black", font=("Helvetica", 12))
        self.metrics_button.pack(pady=10)
    def set_result(self, diagnosis_result):
        result_text = "Positive" if diagnosis_result == 1 else "Negative"
        self.result_label.config(text=result_text)
        color = "#FF3B30" if diagnosis_result == 1 else "#34C759"  # Red for positive, green for negative
        self.result_label.config(fg=color)


    def save_to_file(self):
        """
        Save the results to an Excel file.
        """
        name, surname, date = self.first_page_reference.get_data()
        diagnosis_result = self.result_label.cget("text")
        documents_folder = Path.home() / 'Documents'
        file_path = documents_folder / 'diagnosis_results.xlsx'

        try:
            # Create DataFrame for new data
            new_data = pd.DataFrame({
                "Patient Name": [name],
                "Patient Surname": [surname],
                "Examination Date": [date],
                "Diagnosis Result": [diagnosis_result]
            })

            # Check if file exists to append or create new
            if os.path.exists(file_path):
                # Load existing data
                try:
                    existing_data = pd.read_excel(file_path)
                    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                except Exception as e:
                    messagebox.showerror("Error", f"Error reading existing file: {str(e)}")
                    return
            else:
                updated_data = new_data

            # Save the updated data
            updated_data.to_excel(file_path, index=False)
            messagebox.showinfo("Save Successful", f"Result added successfully to Excel file at {file_path}")

        except ImportError:
            messagebox.showerror("Error", "The openpyxl module is not installed. Please install it using 'pip install openpyxl'")
        except PermissionError:
            messagebox.showerror("Error", f"Permission denied. Unable to save to {file_path}. The file may be open in another program.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
    def show_metrics(self):
        """
        Display model metrics in a new window.
        """
        # Get the metrics from the model evaluation
        accuracy, precision_M, precision_B, recall_M, specificity, f1_M, f1_B, roc_auc = evaluate_model(app.X_test, app.y_test, app.reg_model)
        results = [accuracy, precision_M, precision_B, recall_M, specificity, f1_M, f1_B, roc_auc]
        labels = ['Accuracy', 'Precision of Malignant', 'Precision of Benign','Sensitivity',
                  'Specificity', 'F1-Score of Malignant', 'F1-Score of Benign', 'ROC - AUC']

        # Create a pie chart for each metric
        fig, ax = plt.subplots(2, 4, figsize=(8, 8))
        fig.suptitle('Metrics', fontsize=12)

        # Plotting each metric in a separate subplot
        for i in range(2):
            for j in range(4):
                index = i * 4 + j
                wedge, text, autotext = ax[i, j].pie(
                    [results[index], 100 - results[index]],
                    labels=[labels[index], ''],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#297aa7', '#2ad5e8']
                )
                ax[i, j].axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

                 # Reduce font size for readability
                for txt in text + autotext:
                    txt.set_fontsize(8)

        # Displaying the metrics in a new window
        metrics_window = Toplevel(self)
        metrics_window.geometry("800x600")
        metrics_window.title("Metrics")

        # Embedding the plot onto the tkinter window
        results_canvas = FigureCanvasTkAgg(fig, master=metrics_window)
        results_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        results_canvas.get_tk_widget().configure(bg='red')
        results_canvas.draw()

class DataEntryPage(Frame):
    """The DataEntryPage class allows the user to manually enter data based on the data category."""

    def __init__(self, app, category):
        Frame.__init__(self, app.root)
        self.app = app
        self.category = category
        self.entries = []
        self.labels = ["Radius", "Texture", "Perimeter", "Area", "Smoothness",
                       "Compactness", "Concavity", "Concave Points", "Symmetry", "Fractal Dimension"]
        self.configure(bg=BACKGROUND_COLOR)
        self.create_widgets()

    def create_widgets(self):
        Label(self, text=f"Enter {self.category} Data", fg=PRIMARY_COLOR, bg=BACKGROUND_COLOR,
              font=("Helvetica", 16, 'bold')).pack(pady=20)

        for label_text in self.labels:
            frame = Frame(self, bg=BACKGROUND_COLOR)
            frame.pack(fill=X, padx=20, pady=5)

            label = Label(frame, text=label_text, fg=TEXT_COLOR, bg=BACKGROUND_COLOR, font=("Helvetica", 12))
            label.pack(side=LEFT)

            entry = Entry(frame, fg=TEXT_COLOR, bg=BACKGROUND_COLOR, insertbackground=TEXT_COLOR,
                          font=("Helvetica", 12))
            entry.pack(side=RIGHT, expand=True, fill=X)
            self.entries.append(entry)

        self.next_button = Button(self, text="Next", command=self.next_page,
                                  bg=PRIMARY_COLOR, fg="black", font=("Helvetica", 12))
        self.next_button.pack(pady=20)

        self.back_button = Button(self, text="Back", command=self.app.show_previous_page,
                                  bg=PRIMARY_COLOR, fg="black", font=("Helvetica", 12))
        self.back_button.pack(pady=10)
    def next_page(self):
        if not self.validate_data():
            return  # Stop if validation fails

        try:
            data = [entry.get() for entry in self.entries]
            print(data)
            self.data_callback(data, self.category)
        except ValueError:
            messagebox.showerror("Error", "Please enter only valid numbers.")
            return

    def load_existing_data(self):
        """
        Loads existing data if available when navigating back to this page.
        """
        category_index = {"Mean": 0, "Standard Error": 1, "Worst Mean": 2}
        index = category_index[self.category]
        if len(self.app.all_data) > index:
            existing_data = self.app.all_data[index]
            for entry, value in zip(self.entries, existing_data):
                entry.delete(0, END)
                entry.insert(0, value)

    def data_callback(self, data, category):
        """
        Handles the data processing after data entry. Updates the data array and navigates to the next data page or result page.

        Args:
            data (list): List of data entered by the user.
            category (str): Category of the entered data.
        """
        category_index = {"Mean": 0, "Standard Error": 1, "Worst Mean": 2}
        index = category_index[category]

        # Ensuring all_data has the correct length
        while len(self.app.all_data) <= index:
            self.app.all_data.append([])

        # Updating the data for the specified category
        self.app.all_data[index] = data

        # Navigating to the next page based on the category
        if category == "Mean":
            self.app.current_page.pack_forget()
            self.app.current_page = self.app.data_page2
            self.app.show_current_page()
        elif category == "Standard Error":
            self.app.current_page.pack_forget()
            self.app.current_page = self.app.data_page3
            self.app.show_current_page()
        elif category == "Worst Mean":
            self.predict_diagnosis2()

    def predict_diagnosis2(self):
        """
        Predicts the diagnosis based on the entered data and displays the result.
        """
        try:
            # Transform all_data into the required format for prediction
            combined_data = [item for sublist in self.app.all_data for item in sublist]
            input_data = pd.DataFrame([combined_data])
            print (input_data)

            # Predict the diagnosis using the regression model
            diagnosis_result = self.app.reg_model.predict(input_data)[0]

            # Navigate to the result page with the diagnosis result
            self.app.show_result_page(diagnosis_result)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while predicting diagnosis: {str(e)}")

    def validate_data(self):
        """
        Validates the data entered by the user.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        for i, entry in enumerate(self.entries):
            label = self.labels[i]
            value = entry.get().strip()

            if not value:
                messagebox.showerror("Invalid Input", f"Please enter a value for {label}.")
                return False

            try:
                value = float(value)
                if value == 0:
                    messagebox.showerror("Invalid Input", f"Value for {label} cannot be zero.")
                    return False
            except ValueError:
                messagebox.showerror("Invalid Input", f"Please enter a valid number for {label}.")
                return False

            # Retrieve bounds for the specific feature
            bounds = self.app.category_bounds[self.category][label]
            if not (bounds[0] <= value <= bounds[1]):
                response = messagebox.askyesno(
                    "Value Out of Bounds",
                    f"The value {value} for {label} may be out of typical bounds.\n"
                    "Do you want to continue anyway?"
                )
                if not response:
                    return False

        return True


    def reset_entries(self):
        """Clears the input fields."""
        for entry in self.entries:
            entry.delete(0, END)

class MyApp:
    """Class responsible for managing the main user interface and handling pages."""

    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Prediction")
        self.root.geometry("2000x800")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.all_data = []
        self.reg_model = load_model()

        self.style = Style()
        self.style.theme_use('clam')
        self.style.configure('.', background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=("Helvetica", 12))
        self.style.configure('TButton', background=PRIMARY_COLOR, foreground=BACKGROUND_COLOR)
        self.style.map('TButton', background=[('active', SECONDARY_COLOR)])

        self.root.configure(bg=BACKGROUND_COLOR)
        # Setting up bounds for data validation
        self.category_bounds = {
            "Mean": {
                "Radius": (5.58, 24.37),
                "Texture": (7.73, 30.24),
                "Perimeter": (31.78, 165.72),
                "Area": (215.66, 1786.60),
                "Smoothness": (0.06, 0.13),
                "Compactness": (0.03, 0.28),
                "Concavity": (0.01, 0.35),
                "Concave Points": (0.01, 0.16),
                "Symmetry": (0.11, 0.26),
                "Fractal Dimension": (0.05, 0.09),
            },
            "Standard Error": {
                "Radius": (0.12, 1.29),
                "Texture": (0.41, 2.92),
                "Perimeter": (0.95, 9.69),
                "Area": (8.51, 177.68),
                "Smoothness": (0.001, 0.02),
                "Compactness": (0.0047, 0.09),
                "Concavity": (0.01, 0.12),
                "Concave Points": (0.001, 0.03),
                "Symmetry": (0.01, 0.05),
                "Fractal Dimension": (0.001, 0.013),
            },
            "Worst Mean": {
                "Radius": (4.34, 30.76),
                "Texture": (8.12, 42.68),
                "Perimeter": (22.17, 208.30),
                "Area": (256.19, 2918.16),
                "Smoothness": (0.07, 0.19),
                "Compactness": (0.05, 0.78),
                "Concavity": (0.01, 0.90),
                "Concave Points": (0.01, 0.31),
                "Symmetry": (0.15, 0.49),
                "Fractal Dimension": (0.04, 0.14),
            }
        }
        # Mapping CSV column names to their respective labels and categories
        self.label_to_csv_column = {
            "radius_mean": ("Radius", "Mean"),
            "texture_mean": ("Texture", "Mean"),
            "perimeter_mean": ("Perimeter", "Mean"),
            "area_mean": ("Area", "Mean"),
            "smoothness_mean": ("Smoothness", "Mean"),
            "compactness_mean": ("Compactness", "Mean"),
            "concavity_mean": ("Concavity", "Mean"),
            "concave points_mean": ("Concave Points", "Mean"),
            "symmetry_mean": ("Symmetry", "Mean"),
            "fractal_dimension_mean": ("Fractal Dimension", "Mean"),
            "radius_se": ("Radius", "Standard Error"),
            "texture_se": ("Texture", "Standard Error"),
            "perimeter_se": ("Perimeter", "Standard Error"),
            "area_se": ("Area", "Standard Error"),
            "smoothness_se": ("Smoothness", "Standard Error"),
            "compactness_se": ("Compactness", "Standard Error"),
            "concavity_se": ("Concavity", "Standard Error"),
            "concave points_se": ("Concave Points", "Standard Error"),
            "symmetry_se": ("Symmetry", "Standard Error"),
            "fractal_dimension_se": ("Fractal Dimension", "Standard Error"),
            "radius_worst": ("Radius", "Worst Mean"),
            "texture_worst": ("Texture", "Worst Mean"),
            "perimeter_worst": ("Perimeter", "Worst Mean"),
            "area_worst": ("Area", "Worst Mean"),
            "smoothness_worst": ("Smoothness", "Worst Mean"),
            "compactness_worst": ("Compactness", "Worst Mean"),
            "concavity_worst": ("Concavity", "Worst Mean"),
            "concave points_worst": ("Concave Points", "Worst Mean"),
            "symmetry_worst": ("Symmetry", "Worst Mean"),
            "fractal_dimension_worst": ("Fractal Dimension", "Worst Mean"),
        }

        try:
             # Attempt to load the machine learning model along with test data
            (self.reg_model, self.X_test, self.y_test) = load_model()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model data: {str(e)}")

        self.root.configure(bg="black")

        self.page1 = FirstPage(self)
        self.result_page = ResultPage(self, self.page1)
        self.data_page1 = DataEntryPage(self, "Mean")
        self.data_page2 = DataEntryPage(self, "Standard Error")
        self.data_page3 = DataEntryPage(self, "Worst Mean")

        self.current_page = self.page1
        self.show_current_page()

    def on_closing(self):
        """Handles the window closing event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()

    def show_result_page(self, diagnosis_result):
        """Displays the result page with the diagnosis result.

        Args:
            diagnosis_result: The diagnosis result to be displayed.
        """
        self.result_page.set_result(diagnosis_result)
        self.current_page.pack_forget()
        self.current_page = self.result_page
        self.show_current_page()

    def show_data_page1(self):
        """Displays the first data entry page."""
        self.current_page.pack_forget()
        self.current_page = self.data_page1
        self.show_current_page()

    def show_current_page(self):
        """Helper method to display the current page."""
        self.current_page.pack()

    def show_previous_page(self):
        """Navigates to the previous page based on the current context."""
        if self.current_page == self.data_page1 or self.current_page ==self.result_page:
            self.page1.reset_entries()
            self.data_page1.reset_entries()
            self.data_page2.reset_entries()
            self.data_page3.reset_entries()
            self.current_page.pack_forget()
            self.current_page = self.page1
        elif self.current_page == self.data_page2:
            self.current_page.pack_forget()
            self.current_page = self.data_page1
            self.current_page.load_existing_data()
        elif self.current_page == self.data_page3:
            self.current_page.pack_forget()
            self.current_page = self.data_page2
            self.current_page.load_existing_data()

        self.show_current_page()


if __name__ == "__main__":
    root = Tk()
    app = MyApp(root)
    root.mainloop()
