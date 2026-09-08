import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
import pandas as pd

# ==========================================================
# Task 1: Create Arrow Table
# ==========================================================

data = {
    "employee_id": [1, 2, 3, 4, 5, 6],
    "name": ["Asha", "Rahul", "Neha", "Vikram", "Priya", "Arjun"],
    "department": ["IT", "HR", "IT", "Finance", "HR", "Finance"],
    "salary": [60000, 45000, 70000, 55000, 48000, 65000],
    "city": ["Delhi", "Mumbai", "Bengaluru", "Delhi", "Mumbai", "Chennai"]
}

employee_table = pa.table(data)

print("=" * 50)
print("Arrow Table")
print("=" * 50)
print(employee_table)

# ==========================================================
# Task 2: Display Schema
# ==========================================================

print("\nSchema")
print(employee_table.schema)

print("\nData Types")
print("employee_id:", employee_table.schema.field("employee_id").type)
print("name:", employee_table.schema.field("name").type)
print("salary:", employee_table.schema.field("salary").type)

# ==========================================================
# Task 3: Inspect Table
# ==========================================================

print("\nRows:", employee_table.num_rows)
print("Columns:", employee_table.num_columns)
print("Column Names:", employee_table.column_names)

print("\nName Column")
print(employee_table.column("name"))

print("\nFirst Three Rows")
print(employee_table.slice(0, 3))

# ==========================================================
# Task 4: Select Columns
# ==========================================================

selected_table = employee_table.select(
    ["name", "department", "salary"]
)

print("\nSelected Columns")
print(selected_table)

# ==========================================================
# Task 5: Filter Salary > 50000
# ==========================================================

salary_filter = pc.greater(
    employee_table["salary"],
    50000
)

high_salary = employee_table.filter(salary_filter)

print("\nHigh Salary Employees")
print(high_salary)

# ==========================================================
# Task 6: Filter IT Department
# ==========================================================

department_filter = pc.equal(
    employee_table["department"],
    "IT"
)

it_table = employee_table.filter(department_filter)

print("\nIT Employees")
print(it_table)

# ==========================================================
# Task 7: Calculations
# ==========================================================

salary = employee_table["salary"]

print("\nStatistics")
print("Average Salary:", pc.mean(salary).as_py())
print("Maximum Salary:", pc.max(salary).as_py())
print("Minimum Salary:", pc.min(salary).as_py())
print("Total Salary:", pc.sum(salary).as_py())

# ==========================================================
# Task 8: Add Bonus Column
# ==========================================================

bonus = pc.multiply(employee_table["salary"], 0.10)

employee_table = employee_table.append_column(
    "bonus",
    bonus
)

print("\nTable with Bonus")
print(employee_table)

# ==========================================================
# Task 9: Arrow to Pandas
# ==========================================================

employee_df = employee_table.to_pandas()

print("\nPandas DataFrame")
print(employee_df)

# ==========================================================
# Task 10: Pandas to Arrow
# ==========================================================

new_arrow_table = pa.Table.from_pandas(
    employee_df,
    preserve_index=False
)

print("\nConverted Back to Arrow")
print(new_arrow_table)

# ==========================================================
# Task 11: Save as Parquet
# ==========================================================

pq.write_table(
    employee_table,
    "employees.parquet"
)

print("\nemployees.parquet created.")

# ==========================================================
# Task 12: Read Parquet
# ==========================================================

loaded_table = pq.read_table(
    "employees.parquet"
)

print("\nLoaded Parquet")
print(loaded_table)

# ==========================================================
# Task 13: Save as Arrow IPC
# ==========================================================

with ipc.new_file(
    "employees.arrow",
    employee_table.schema
) as writer:
    writer.write_table(employee_table)

print("\nemployees.arrow created.")

# ==========================================================
# Task 14: Read Arrow IPC
# ==========================================================

with ipc.open_file("employees.arrow") as reader:
    ipc_table = reader.read_all()

print("\nLoaded Arrow File")
print(ipc_table)

# ==========================================================
# Bonus 1: Employees from Delhi
# ==========================================================

delhi_filter = pc.equal(
    employee_table["city"],
    "Delhi"
)

print("\nEmployees from Delhi")
print(employee_table.filter(delhi_filter))

# ==========================================================
# Bonus 2: Salary Between 50000 and 65000
# ==========================================================

salary_range = pc.and_(
    pc.greater_equal(employee_table["salary"], 50000),
    pc.less_equal(employee_table["salary"], 65000)
)

print("\nSalary Between 50000 and 65000")
print(employee_table.filter(salary_range))

# ==========================================================
# Bonus 3: Annual Salary
# ==========================================================

annual_salary = pc.multiply(
    employee_table["salary"],
    12
)

employee_table = employee_table.append_column(
    "annual_salary",
    annual_salary
)

print("\nTable with Annual Salary")
print(employee_table)

# ==========================================================
# Bonus 4: Save IT Employees
# ==========================================================

pq.write_table(
    it_table,
    "it_employees.parquet"
)

print("\nit_employees.parquet created.")

# ==========================================================
# Bonus 5: Read Selected Columns
# ==========================================================

selected_columns = pq.read_table(
    "employees.parquet",
    columns=["name", "salary"]
)

print("\nSelected Columns from Parquet")
print(selected_columns)

# ==========================================================
# Bonus 6: Sort by Salary
# ==========================================================

sorted_df = employee_table.to_pandas().sort_values(
    by="salary",
    ascending=False
)

print("\nSorted by Salary")
print(sorted_df)