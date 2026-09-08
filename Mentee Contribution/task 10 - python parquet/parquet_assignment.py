import pandas as pd

# Task 1: Create DataFrame
data = {
    "employee_id": [1, 2, 3, 4, 5],
    "name": ["Asha", "Rahul", "Neha", "Vikram", "Priya"],
    "department": ["IT", "HR", "IT", "Finance", "HR"],
    "salary": [60000, 45000, 70000, 55000, 48000]
}

employees_df = pd.DataFrame(data)

print("Employee Records")
print(employees_df)

# Task 2: Save DataFrame as Parquet
employees_df.to_parquet(
    "employees.parquet",
    engine="pyarrow",
    index=False
)

# Task 3: Read Parquet File
loaded_df = pd.read_parquet(
    "employees.parquet",
    engine="pyarrow"
)

print("\nLoaded Data")
print(loaded_df)

# Task 4: Analysis

# Employees with salary > 50000
high_salary = loaded_df[loaded_df["salary"] > 50000]

print("\nEmployees with Salary > 50000")
print(high_salary)

# Average salary
average_salary = loaded_df["salary"].mean()
print("\nAverage Salary:", average_salary)

# Department count
dept_count = loaded_df["department"].value_counts()
print("\nEmployees by Department")
print(dept_count)

# Task 5: Save filtered data
high_salary.to_parquet(
    "high_salary_employees.parquet",
    engine="pyarrow",
    index=False
)

# Bonus Task
bonus_df = pd.read_parquet(
    "employees.parquet",
    columns=["name", "salary"]
)

print("\nBonus Task")
print(bonus_df)