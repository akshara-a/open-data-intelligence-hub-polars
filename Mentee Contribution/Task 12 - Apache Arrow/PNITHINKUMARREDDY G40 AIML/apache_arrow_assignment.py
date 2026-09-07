import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
import pandas as pd


# -----------------------------------------------------
# Employee Dataset
# -----------------------------------------------------

employee_data = {
    "employee_id": [1, 2, 3, 4, 5, 6],
    "name": ["Asha", "Rahul", "Neha", "Vikram", "Priya", "Arjun"],
    "department": ["IT", "HR", "IT", "Finance", "HR", "Finance"],
    "salary": [60000, 45000, 70000, 55000, 48000, 65000],
    "city": ["Delhi", "Mumbai", "Bengaluru", "Delhi", "Mumbai", "Chennai"]
}

employees = pa.table(employee_data)

print("\nArrow Table")
print(employees)


# -----------------------------------------------------
# Schema Information
# -----------------------------------------------------

print("\nSchema")
print(employees.schema)

print("\nSchema Answers")
print("employee_id :", employees.schema.field("employee_id").type)
print("name        :", employees.schema.field("name").type)
print("salary      :", employees.schema.field("salary").type)


# -----------------------------------------------------
# Table Overview
# -----------------------------------------------------

print("\nTable Information")
print("Rows:", employees.num_rows)
print("Columns:", employees.num_columns)
print("Column Names:", employees.column_names)

print("\nName Column")
print(employees.column("name"))

print("\nFirst Three Records")
print(employees.slice(0, 3))


# -----------------------------------------------------
# Selecting Required Columns
# -----------------------------------------------------

employee_summary = employees.select(
    ["name", "department", "salary"]
)

print("\nSelected Columns")
print(employee_summary)


# -----------------------------------------------------
# Employees with Salary Greater Than 50000
# -----------------------------------------------------

salary_condition = pc.greater(
    employees["salary"],
    50000
)

high_salary_employees = employees.filter(salary_condition)

print("\nEmployees with Salary > 50000")
print(high_salary_employees)


# -----------------------------------------------------
# Employees from IT Department
# -----------------------------------------------------

it_condition = pc.equal(
    employees["department"],
    "IT"
)

it_employees = employees.filter(it_condition)

print("\nIT Employees")
print(it_employees)


# -----------------------------------------------------
# Salary Statistics
# -----------------------------------------------------

salary = employees["salary"]

print("\nSalary Statistics")
print("Average Salary :", pc.mean(salary).as_py())
print("Maximum Salary :", pc.max(salary).as_py())
print("Minimum Salary :", pc.min(salary).as_py())
print("Total Salary   :", pc.sum(salary).as_py())


# -----------------------------------------------------
# Adding Bonus Column
# -----------------------------------------------------

bonus = pc.multiply(
    employees["salary"],
    0.10
)

employees = employees.append_column(
    "bonus",
    bonus
)

print("\nTable with Bonus")
print(employees)


# -----------------------------------------------------
# Arrow to Pandas
# -----------------------------------------------------

employee_df = employees.to_pandas()

print("\nPandas DataFrame")
print(employee_df)


# -----------------------------------------------------
# Pandas to Arrow
# -----------------------------------------------------

arrow_from_dataframe = pa.Table.from_pandas(
    employee_df,
    preserve_index=False
)

print("\nArrow Table from Pandas")
print(arrow_from_dataframe)


# -----------------------------------------------------
# Save as Parquet
# -----------------------------------------------------

pq.write_table(
    employees,
    "employees.parquet"
)

print("\nParquet file created successfully.")


# -----------------------------------------------------
# Read Parquet File
# -----------------------------------------------------

parquet_table = pq.read_table(
    "employees.parquet"
)

print("\nParquet File Contents")
print(parquet_table)


# -----------------------------------------------------
# Save as Arrow IPC File
# -----------------------------------------------------

with ipc.new_file(
    "employees.arrow",
    employees.schema
) as writer:
    writer.write_table(employees)

print("\nArrow IPC file created successfully.")


# -----------------------------------------------------
# Read Arrow IPC File
# -----------------------------------------------------

with ipc.open_file("employees.arrow") as reader:
    arrow_table = reader.read_all()

print("\nArrow IPC File Contents")
print(arrow_table)


# -----------------------------------------------------
# Additional Operations
# -----------------------------------------------------

print("\nEmployees Working in Delhi")

delhi_employees = employees.filter(
    pc.equal(employees["city"], "Delhi")
)

print(delhi_employees)


print("\nEmployees with Salary Between 50000 and 65000")

salary_range = pc.and_(
    pc.greater_equal(employees["salary"], 50000),
    pc.less_equal(employees["salary"], 65000)
)

mid_salary_employees = employees.filter(salary_range)

print(mid_salary_employees)


annual_salary = pc.multiply(
    employees["salary"],
    12
)

employees = employees.append_column(
    "annual_salary",
    annual_salary
)

print("\nTable with Annual Salary")
print(employees)


pq.write_table(
    it_employees,
    "it_employees.parquet"
)

print("\nIT employees saved successfully.")


selected_data = pq.read_table(
    "employees.parquet",
    columns=["name", "salary"]
)

print("\nSelected Columns from Parquet")
print(selected_data)


sorted_indices = pc.sort_indices(
    employees,
    sort_keys=[("salary", "descending")]
)

sorted_employees = employees.take(sorted_indices)

print("\nEmployees Sorted by Salary")
print(sorted_employees)