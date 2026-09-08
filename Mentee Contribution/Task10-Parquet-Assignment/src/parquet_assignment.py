import pandas as pd
from pathlib import Path

# Get the project folder path
BASE_DIR = Path(__file__).resolve().parent.parent

# Define file paths
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

EMPLOYEES_FILE = DATA_DIR / "employees.parquet"
HIGH_SALARY_FILE = OUTPUTS_DIR / "high_salary_employees.parquet"

# Create folders if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Task 1: Create the DataFrame
data = {
    "employee_id": [1, 2, 3, 4, 5],
    "name": ["Asha", "Rahul", "Neha", "Vikram", "Priya"],
    "department": ["IT", "HR", "IT", "Finance", "HR"],
    "salary": [60000, 45000, 70000, 55000, 48000]
}

employees_df = pd.DataFrame(data)

print("All Employee Records:")
print(employees_df)

# Task 2: Save the DataFrame as a Parquet file
employees_df.to_parquet(EMPLOYEES_FILE, index=False)

print("\nemployees.parquet file created successfully.")

# Task 3: Read the Parquet file
loaded_df = pd.read_parquet(EMPLOYEES_FILE)

print("\nEmployees Read from Parquet File:")
print(loaded_df)

# Task 4.1: Display employees with salary greater than 50000
high_salary_df = loaded_df[loaded_df["salary"] > 50000]

print("\nEmployees with Salary Greater Than 50000:")
print(high_salary_df)

# Task 4.2: Calculate the average salary
average_salary = loaded_df["salary"].mean()

print("\nAverage Salary:")
print(average_salary)

# Task 4.3: Display the number of employees in each department
department_counts = loaded_df["department"].value_counts()

print("\nNumber of Employees in Each Department:")
print(department_counts)

# Task 5: Save filtered data as Parquet
high_salary_df.to_parquet(HIGH_SALARY_FILE, index=False)

print("\nhigh_salary_employees.parquet file created successfully.")

# Bonus Task: Read only name and salary columns
bonus_df = pd.read_parquet(
    EMPLOYEES_FILE,
    columns=["name", "salary"]
)

print("\nBonus - Name and Salary Only:")
print(bonus_df)