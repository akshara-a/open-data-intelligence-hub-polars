import pandas as pd

# -------------------------------------
# Task 1: Create DataFrame
# -------------------------------------

data = {
    "employee_id": [1, 2, 3, 4, 5],
    "name": ["Asha", "Rahul", "Neha", "Vikram", "Priya"],
    "department": ["IT", "HR", "IT", "Finance", "HR"],
    "salary": [60000, 45000, 70000, 55000, 48000]
}

employees_df = pd.DataFrame(data)

print("All Employee Records")
print(employees_df)

# -------------------------------------
# Task 2: Save DataFrame as Parquet
# -------------------------------------

employees_df.to_parquet("employees.parquet", index=False)

print("\nemployees.parquet file created successfully.")

# -------------------------------------
# Task 3: Read Parquet File
# -------------------------------------

loaded_df = pd.read_parquet("employees.parquet")

print("\nData Read From Parquet File")
print(loaded_df)

# -------------------------------------
# Task 4A: Employees earning more than 50000
# -------------------------------------

high_salary = loaded_df[loaded_df["salary"] > 50000]

print("\nEmployees with Salary Greater Than 50000")
print(high_salary)

# -------------------------------------
# Task 4B: Average Salary
# -------------------------------------

average_salary = loaded_df["salary"].mean()

print("\nAverage Salary:")
print(average_salary)

# -------------------------------------
# Task 4C: Employee Count by Department
# -------------------------------------

department_count = loaded_df["department"].value_counts()

print("\nEmployee Count by Department")
print(department_count)

# -------------------------------------
# Task 5: Save High Salary Employees
# -------------------------------------

high_salary.to_parquet("high_salary_employees.parquet", index=False)

print("\nhigh_salary_employees.parquet created successfully.")

# -------------------------------------
# Bonus Task
# -------------------------------------

selected_columns = pd.read_parquet(
    "employees.parquet",
    columns=["name", "salary"]
)

print("\nOnly Name and Salary Columns")
print(selected_columns)