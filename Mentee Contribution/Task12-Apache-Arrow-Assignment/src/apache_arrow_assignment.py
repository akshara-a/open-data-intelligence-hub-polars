import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.ipc as ipc


# ============================================================
# Task 1: Create an Arrow Table
# ============================================================

data = {
    "employee_id": [1, 2, 3, 4, 5, 6],
    "name": ["Asha", "Rahul", "Neha", "Vikram", "Priya", "Arjun"],
    "department": ["IT", "HR", "IT", "Finance", "HR", "Finance"],
    "salary": [60000, 45000, 70000, 55000, 48000, 65000],
    "city": ["Delhi", "Mumbai", "Bengaluru", "Delhi", "Mumbai", "Chennai"]
}

employee_table = pa.table(data)

print("=" * 60)
print("TASK 1: ARROW TABLE")
print("=" * 60)
print(employee_table)


# ============================================================
# Task 2: Display the Schema
# ============================================================

print("\n" + "=" * 60)
print("TASK 2: SCHEMA")
print("=" * 60)
print(employee_table.schema)

print("\nData Types:")
print("employee_id:", employee_table.schema.field("employee_id").type)
print("name:", employee_table.schema.field("name").type)
print("salary:", employee_table.schema.field("salary").type)


# ============================================================
# Task 3: Inspect the Table
# ============================================================

print("\n" + "=" * 60)
print("TASK 3: TABLE INSPECTION")
print("=" * 60)

print("Rows:", employee_table.num_rows)
print("Columns:", employee_table.num_columns)
print("Column names:", employee_table.column_names)

print("\nName column:")
print(employee_table.column("name"))

print("\nFirst three rows:")
print(employee_table.slice(0, 3))


# ============================================================
# Task 4: Select Specific Columns
# ============================================================

selected_table = employee_table.select(
    ["name", "department", "salary"]
)

print("\n" + "=" * 60)
print("TASK 4: SELECTED COLUMNS")
print("=" * 60)
print(selected_table)


# ============================================================
# Task 5: Filter Salary > 50000
# ============================================================

salary_filter = pc.greater(
    employee_table["salary"],
    50000
)

high_salary_table = employee_table.filter(salary_filter)

print("\n" + "=" * 60)
print("TASK 5: SALARY GREATER THAN 50000")
print("=" * 60)
print(high_salary_table)


# ============================================================
# Task 6: Filter IT Department
# ============================================================

department_filter = pc.equal(
    employee_table["department"],
    "IT"
)

it_employees = employee_table.filter(department_filter)

print("\n" + "=" * 60)
print("TASK 6: IT EMPLOYEES")
print("=" * 60)
print(it_employees)


# ============================================================
# Task 7: Salary Calculations
# ============================================================

salary_column = employee_table["salary"]

average_salary = pc.mean(salary_column).as_py()
maximum_salary = pc.max(salary_column).as_py()
minimum_salary = pc.min(salary_column).as_py()
total_salary = pc.sum(salary_column).as_py()

print("\n" + "=" * 60)
print("TASK 7: SALARY CALCULATIONS")
print("=" * 60)

print("Average salary:", average_salary)
print("Maximum salary:", maximum_salary)
print("Minimum salary:", minimum_salary)
print("Total salary:", total_salary)


# ============================================================
# Task 8: Add Bonus Column
# ============================================================

bonus_column = pc.multiply(
    employee_table["salary"],
    0.10
)

employee_table = employee_table.append_column(
    "bonus",
    bonus_column
)

print("\n" + "=" * 60)
print("TASK 8: BONUS COLUMN")
print("=" * 60)
print(employee_table)


# ============================================================
# Task 9: Convert Arrow to Pandas
# ============================================================

employee_df = employee_table.to_pandas()

print("\n" + "=" * 60)
print("TASK 9: ARROW TO PANDAS")
print("=" * 60)
print(employee_df)


# ============================================================
# Task 10: Convert Pandas to Arrow
# ============================================================

new_arrow_table = pa.Table.from_pandas(
    employee_df,
    preserve_index=False
)

print("\n" + "=" * 60)
print("TASK 10: PANDAS TO ARROW")
print("=" * 60)
print(new_arrow_table)


# ============================================================
# Task 11: Save as Parquet
# ============================================================

pq.write_table(
    employee_table,
    "data/employees.parquet"
)

print("\n" + "=" * 60)
print("TASK 11: PARQUET FILE")
print("=" * 60)
print("data/employees.parquet created successfully.")


# ============================================================
# Task 12: Read Parquet File
# ============================================================

loaded_table = pq.read_table(
    "data/employees.parquet"
)

print("\n" + "=" * 60)
print("TASK 12: READ PARQUET")
print("=" * 60)
print(loaded_table)


# ============================================================
# Task 13: Save as Arrow IPC File
# ============================================================

with ipc.new_file(
    "data/employees.arrow",
    employee_table.schema
) as writer:
    writer.write_table(employee_table)

print("\n" + "=" * 60)
print("TASK 13: ARROW IPC FILE")
print("=" * 60)
print("data/employees.arrow created successfully.")


# ============================================================
# Task 14: Read Arrow IPC File
# ============================================================

with ipc.open_file("data/employees.arrow") as reader:
    ipc_table = reader.read_all()

print("\n" + "=" * 60)
print("TASK 14: READ ARROW IPC FILE")
print("=" * 60)
print(ipc_table)


# ============================================================
# BONUS 1: Employees from Delhi
# ============================================================

delhi_filter = pc.equal(
    employee_table["city"],
    "Delhi"
)

delhi_employees = employee_table.filter(delhi_filter)

print("\n" + "=" * 60)
print("BONUS 1: DELHI EMPLOYEES")
print("=" * 60)
print(delhi_employees)


# ============================================================
# BONUS 2: Salary Between 50000 and 65000
# ============================================================

salary_min = pc.greater_equal(
    employee_table["salary"],
    50000
)

salary_max = pc.less_equal(
    employee_table["salary"],
    65000
)

salary_range_filter = pc.and_(
    salary_min,
    salary_max
)

salary_range_employees = employee_table.filter(
    salary_range_filter
)

print("\n" + "=" * 60)
print("BONUS 2: SALARY BETWEEN 50000 AND 65000")
print("=" * 60)
print(salary_range_employees)


# ============================================================
# BONUS 3: Add Annual Salary Column
# ============================================================

annual_salary_column = pc.multiply(
    employee_table["salary"],
    12
)

employee_with_annual_salary = employee_table.append_column(
    "annual_salary",
    annual_salary_column
)

print("\n" + "=" * 60)
print("BONUS 3: ANNUAL SALARY")
print("=" * 60)
print(employee_with_annual_salary)


# ============================================================
# BONUS 4: Save IT Employees
# ============================================================

pq.write_table(
    it_employees,
    "outputs/it_employees.parquet"
)

print("\n" + "=" * 60)
print("BONUS 4: IT EMPLOYEES PARQUET")
print("=" * 60)
print("outputs/it_employees.parquet created successfully.")


# ============================================================
# BONUS 5: Read Only Name and Salary Columns
# ============================================================

selected_columns = pq.read_table(
    "data/employees.parquet",
    columns=["name", "salary"]
)

print("\n" + "=" * 60)
print("BONUS 5: NAME AND SALARY ONLY")
print("=" * 60)
print(selected_columns)


# ============================================================
# BONUS 6: Sort Employees by Salary Descending
# ============================================================

sort_indices = pc.sort_indices(
    employee_table,
    sort_keys=[("salary", "descending")]
)

sorted_employees = pc.take(
    employee_table,
    sort_indices
)

print("\n" + "=" * 60)
print("BONUS 6: SORTED BY SALARY")
print("=" * 60)
print(sorted_employees)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("TASK 12 COMPLETED SUCCESSFULLY!")
print("=" * 60)