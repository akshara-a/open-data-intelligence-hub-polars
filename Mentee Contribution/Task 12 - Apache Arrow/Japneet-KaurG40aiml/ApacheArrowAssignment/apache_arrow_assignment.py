# ============================================
# Apache Arrow Assignment
# Name:
# Roll No:
# ============================================

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
import pandas as pd

# ======================================================
# Task 1: Create an Arrow Table
# ======================================================

data = {
    "employee_id": [1, 2, 3, 4, 5, 6],
    "name": ["Asha", "Rahul", "Neha", "Vikram", "Priya", "Arjun"],
    "department": ["IT", "HR", "IT", "Finance", "HR", "Finance"],
    "salary": [60000, 45000, 70000, 55000, 48000, 65000],
    "city": ["Delhi", "Mumbai", "Bengaluru", "Delhi", "Mumbai", "Chennai"]
}

employee_table = pa.table(data)

print("=" * 50)
print("Task 1: Employee Table")
print(employee_table)

# ======================================================
# Task 2: Display Schema
# ======================================================

print("\n" + "=" * 50)
print("Task 2: Schema")
print(employee_table.schema)

print("\nAnswers:")
print("employee_id datatype :", employee_table.schema.field("employee_id").type)
print("name datatype        :", employee_table.schema.field("name").type)
print("salary datatype      :", employee_table.schema.field("salary").type)

# ======================================================
# Task 3: Inspect Table
# ======================================================

print("\n" + "=" * 50)
print("Task 3: Inspect Table")

print("Rows:", employee_table.num_rows)
print("Columns:", employee_table.num_columns)
print("Column Names:", employee_table.column_names)

print("\nName Column")
print(employee_table.column("name"))

print("\nFirst 3 Rows")
print(employee_table.slice(0, 3))

# ======================================================
# Task 4: Select Specific Columns
# ======================================================

print("\n" + "=" * 50)
print("Task 4: Selected Columns")

selected_table = employee_table.select(
    ["name", "department", "salary"]
)

print(selected_table)

# ======================================================
# Task 5: Filter Salary > 50000
# ======================================================

print("\n" + "=" * 50)
print("Task 5: Salary > 50000")

salary_filter = pc.greater(
    employee_table["salary"],
    50000
)

high_salary_table = employee_table.filter(salary_filter)

print(high_salary_table)

# ======================================================
# Task 6: Filter IT Department
# ======================================================

print("\n" + "=" * 50)
print("Task 6: IT Employees")

department_filter = pc.equal(
    employee_table["department"],
    "IT"
)

it_employees = employee_table.filter(department_filter)

print(it_employees)

# ======================================================
# Task 7: Calculations
# ======================================================

print("\n" + "=" * 50)
print("Task 7: Salary Calculations")

salary_column = employee_table["salary"]

print("Average Salary :", pc.mean(salary_column).as_py())
print("Maximum Salary :", pc.max(salary_column).as_py())
print("Minimum Salary :", pc.min(salary_column).as_py())
print("Total Salary   :", pc.sum(salary_column).as_py())

# ======================================================
# Task 8: Add Bonus Column
# ======================================================

print("\n" + "=" * 50)
print("Task 8: Add Bonus Column")

bonus_column = pc.multiply(
    employee_table["salary"],
    0.10
)

employee_table = employee_table.append_column(
    "bonus",
    bonus_column
)

print(employee_table)

# ======================================================
# Task 9: Arrow to Pandas
# ======================================================

print("\n" + "=" * 50)
print("Task 9: Arrow -> Pandas")

employee_df = employee_table.to_pandas()

print(employee_df)

# ======================================================
# Task 10: Pandas to Arrow
# ======================================================

print("\n" + "=" * 50)
print("Task 10: Pandas -> Arrow")

new_arrow_table = pa.Table.from_pandas(
    employee_df,
    preserve_index=False
)

print(new_arrow_table)

# ======================================================
# Task 11: Save Parquet
# ======================================================

print("\n" + "=" * 50)
print("Task 11: Save Parquet")

pq.write_table(
    employee_table,
    "employees.parquet"
)

print("employees.parquet created successfully.")

# ======================================================
# Task 12: Read Parquet
# ======================================================

print("\n" + "=" * 50)
print("Task 12: Read Parquet")

loaded_table = pq.read_table(
    "employees.parquet"
)

print(loaded_table)

# ======================================================
# Task 13: Save Arrow IPC
# ======================================================

print("\n" + "=" * 50)
print("Task 13: Save Arrow IPC")

with ipc.new_file(
    "employees.arrow",
    employee_table.schema
) as writer:

    writer.write_table(employee_table)

print("employees.arrow created successfully.")

# ======================================================
# Task 14: Read Arrow IPC
# ======================================================

print("\n" + "=" * 50)
print("Task 14: Read Arrow IPC")

with ipc.open_file(
    "employees.arrow"
) as reader:

    ipc_table = reader.read_all()

print(ipc_table)

# ======================================================
# Bonus Task 1: Employees in Delhi
# ======================================================

print("\n" + "=" * 50)
print("Bonus 1: Employees in Delhi")

delhi_filter = pc.equal(
    employee_table["city"],
    "Delhi"
)

print(employee_table.filter(delhi_filter))

# ======================================================
# Bonus Task 2: Salary Between 50000 and 65000
# ======================================================

print("\n" + "=" * 50)
print("Bonus 2: Salary Between 50000 and 65000")

salary_range = pc.and_(
    pc.greater_equal(employee_table["salary"], 50000),
    pc.less_equal(employee_table["salary"], 65000)
)

print(employee_table.filter(salary_range))

# ======================================================
# Bonus Task 3: Annual Salary
# ======================================================

print("\n" + "=" * 50)
print("Bonus 3: Annual Salary")

annual_salary = pc.multiply(
    employee_table["salary"],
    12
)

employee_table = employee_table.append_column(
    "annual_salary",
    annual_salary
)

print(employee_table)

# ======================================================
# Bonus Task 4: Save IT Employees
# ======================================================

print("\n" + "=" * 50)
print("Bonus 4: Save IT Employees")

pq.write_table(
    it_employees,
    "it_employees.parquet"
)

print("it_employees.parquet created successfully.")

# ======================================================
# Bonus Task 5: Read Selected Columns
# ======================================================

print("\n" + "=" * 50)
print("Bonus 5: Read Name and Salary")

selected_columns = pq.read_table(
    "employees.parquet",
    columns=["name", "salary"]
)

print(selected_columns)

# ======================================================
# Bonus Task 6: Sort by Salary Descending
# ======================================================

print("\n" + "=" * 50)
print("Bonus 6: Salary Highest to Lowest")

sorted_table = employee_table.sort_by(
    [("salary", "descending")]
)

print(sorted_table)

print("\nAssignment Completed Successfully!")