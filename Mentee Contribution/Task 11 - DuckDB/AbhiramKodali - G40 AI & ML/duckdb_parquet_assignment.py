import duckdb
import pandas as pd

# Create employee data
data = {
    "employee_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": ["Asha", "Rahul", "Neha", "Vikram", "Priya", "Arjun", "Kiran", "Sneha"],
    "department": ["IT", "HR", "IT", "Finance", "HR", "IT", "Finance", "HR"],
    "salary": [60000, 45000, 70000, 55000, 48000, 65000, 52000, 58000],
    "city": ["Hyderabad", "Bangalore", "Hyderabad", "Chennai", "Bangalore", "Hyderabad", "Chennai", "Bangalore"]
}

employees_df = pd.DataFrame(data)

print("Employee Data:")
print(employees_df)
# Save employee data as Parquet
employees_df.to_parquet(
    "employees.parquet",
    engine="pyarrow",
    index=False
)

print("\nSaved employees.parquet")

# Connect to DuckDB
con = duckdb.connect("company.duckdb")

print("Connected to DuckDB")

# Query 1: Read all employees from Parquet
result = con.execute("""
    SELECT *
    FROM 'employees.parquet'
""").fetchdf()

print("\nAll Employees:")
print(result)

# Query 2: Employees earning more than 50,000
high_salary = con.execute("""
    SELECT *
    FROM 'employees.parquet'
    WHERE salary > 50000
""").fetchdf()

print("\nEmployees earning more than 50,000:")
print(high_salary)

# Query 3: Employees in the IT department
it_employees = con.execute("""
    SELECT *
    FROM 'employees.parquet'
    WHERE department = 'IT'
""").fetchdf()

print("\nIT Employees:")
print(it_employees)

# Query 4: Employees in Hyderabad
hyderabad_employees = con.execute("""
    SELECT *
    FROM 'employees.parquet'
    WHERE city = 'Hyderabad'
""").fetchdf()

print("\nHyderabad Employees:")
print(hyderabad_employees)

# Query 5: Sort employees by salary descending
sorted_employees = con.execute("""
    SELECT name, department, salary
    FROM 'employees.parquet'
    ORDER BY salary DESC
""").fetchdf()

print("\nEmployees sorted by salary:")
print(sorted_employees)

# Salary analysis
salary_analysis = con.execute("""
    SELECT
        AVG(salary) AS average_salary,
        MIN(salary) AS minimum_salary,
        MAX(salary) AS maximum_salary,
        SUM(salary) AS total_salary
    FROM 'employees.parquet'
""").fetchdf()

print("\nSalary Analysis:")
print(salary_analysis)

# Department-wise employee count and average salary
department_analysis = con.execute("""
    SELECT
        department,
        COUNT(*) AS employee_count,
        AVG(salary) AS average_salary
    FROM 'employees.parquet'
    GROUP BY department
    ORDER BY department
""").fetchdf()

print("\nDepartment Analysis:")
print(department_analysis)

# Create employees table inside DuckDB
con.execute("""
    CREATE OR REPLACE TABLE employees AS
    SELECT *
    FROM 'employees.parquet'
""")

print("\nEmployees table created in company.duckdb")

# Verify the DuckDB table
table_data = con.execute("""
    SELECT *
    FROM employees
""").fetchdf()

print("\nEmployees table:")
print(table_data)

# Export high-salary employees to Parquet
con.execute("""
    COPY (
        SELECT *
        FROM employees
        WHERE salary > 50000
    )
    TO 'high_salary_employees.parquet'
    (FORMAT PARQUET)
""")

print("\nSaved high_salary_employees.parquet")

# Verify exported Parquet file
exported_data = con.execute("""
    SELECT *
    FROM 'high_salary_employees.parquet'
""").fetchdf()

print("\nVerified high-salary employees:")
print(exported_data)

# Bonus: Department with the highest average salary
highest_avg_department = con.execute("""
    SELECT
        department,
        AVG(salary) AS average_salary
    FROM employees
    GROUP BY department
    ORDER BY average_salary DESC
    LIMIT 1
""").fetchdf()

print("\nDepartment with highest average salary:")
print(highest_avg_department)

# Bonus: Employees earning above the overall average salary
above_average = con.execute("""
    SELECT *
    FROM employees
    WHERE salary > (
        SELECT AVG(salary)
        FROM employees
    )
    ORDER BY salary DESC
""").fetchdf()

print("\nEmployees earning above average salary:")
print(above_average)

# Close DuckDB connection
con.close()

print("\nDuckDB connection closed.")