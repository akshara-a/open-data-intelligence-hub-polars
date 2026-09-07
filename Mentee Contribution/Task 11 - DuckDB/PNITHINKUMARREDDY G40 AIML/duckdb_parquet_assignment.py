import pandas as pd
import duckdb

# ==================================================
# Task 1 : Create Employee Data and Save as Parquet
# ==================================================

data = {
    "employee_id": [101,102,103,104,105,106,107,108],
    "name": [
        "Ananya",
        "Rohit",
        "Sneha",
        "Karthik",
        "Divya",
        "Nikhil",
        "Ishita",
        "Varun"
    ],
    "department": [
        "Engineering",
        "Marketing",
        "Finance",
        "Engineering",
        "Human Resources",
        "Finance",
        "Marketing",
        "Sales"
    ],
    "salary": [
        82000,
        56000,
        69000,
        91000,
        48000,
        73000,
        61000,
        54000
    ],
    "city": [
        "Hyderabad",
        "Pune",
        "Chennai",
        "Bengaluru",
        "Kolkata",
        "Hyderabad",
        "Mumbai",
        "Pune"
    ]
}

df = pd.DataFrame(data)

df.to_parquet("employees.parquet", index=False)

print("="*60)
print("Parquet File Created Successfully")
print("="*60)


# ==================================================
# Task 2 : Read Parquet File
# ==================================================

print("\nTASK 2 : Employee Records")

employees = duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
""").df()

print(employees)


# ==================================================
# Task 3 : Filter Records
# ==================================================

print("\nEmployees with Salary > 60000")

print(duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
WHERE salary > 60000
""").df())


print("\nEngineering Department")

print(duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
WHERE department='Engineering'
""").df())


print("\nEmployees from Hyderabad")

print(duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
WHERE city='Hyderabad'
""").df())


print("\nEngineering Employees with Salary > 85000")

print(duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
WHERE department='Engineering'
AND salary > 85000
""").df())


# ==================================================
# Task 4 : Select Columns and Sort
# ==================================================

print("\nName, Department and Salary")

print(duckdb.sql("""
SELECT
name,
department,
salary
FROM read_parquet('employees.parquet')
ORDER BY salary DESC
""").df())


# ==================================================
# Task 5 : Aggregations
# ==================================================

print("\nSalary Summary")

summary = duckdb.sql("""
SELECT
COUNT(*) AS employee_count,
AVG(salary) AS average_salary,
MAX(salary) AS maximum_salary,
MIN(salary) AS minimum_salary,
SUM(salary) AS total_salary
FROM read_parquet('employees.parquet')
""").df()

print(summary)


# ==================================================
# Task 6 : Group By Department
# ==================================================

print("\nDepartment Wise Summary")

department_summary = duckdb.sql("""
SELECT
department,
COUNT(*) AS employee_count,
AVG(salary) AS average_salary,
MAX(salary) AS highest_salary,
SUM(salary) AS total_salary
FROM read_parquet('employees.parquet')
GROUP BY department
ORDER BY average_salary DESC
""").df()

print(department_summary)


# ==================================================
# Task 7 : Create DuckDB Database
# ==================================================

connection = duckdb.connect("company.duckdb")

connection.execute("""
CREATE OR REPLACE TABLE employees AS
SELECT *
FROM read_parquet('employees.parquet')
""")

print("\nEmployees Table in DuckDB")

print(connection.execute("""
SELECT *
FROM employees
""").df())

connection.close()


# ==================================================
# Task 8 : Export High Salary Employees
# ==================================================

duckdb.sql("""
COPY (
SELECT *
FROM read_parquet('employees.parquet')
WHERE salary > 60000
)
TO 'high_salary_employees.parquet'
(FORMAT PARQUET)
""")

print("\nHigh Salary Employees Exported")


# ==================================================
# Task 9 : Verify Export
# ==================================================

print("\nVerification")

print(duckdb.sql("""
SELECT *
FROM read_parquet('high_salary_employees.parquet')
""").df())


# ==================================================
# Bonus Task 1
# Second Highest Salary
# ==================================================

print("\nSecond Highest Salary Employee")

print(duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
ORDER BY salary DESC
LIMIT 1 OFFSET 1
""").df())


# ==================================================
# Bonus Task 2
# Top 3 Salaries
# ==================================================

print("\nTop Three Highest Paid Employees")

print(duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
ORDER BY salary DESC
LIMIT 3
""").df())


# ==================================================
# Bonus Task 3
# Average Salary by City
# ==================================================

print("\nAverage Salary by City")

print(duckdb.sql("""
SELECT
city,
AVG(salary) AS average_salary
FROM read_parquet('employees.parquet')
GROUP BY city
ORDER BY average_salary DESC
""").df())


# ==================================================
# Bonus Task 4
# Departments with Average Salary > 60000
# ==================================================

print("\nDepartments with Average Salary > 60000")

print(duckdb.sql("""
SELECT
department,
AVG(salary) AS average_salary
FROM read_parquet('employees.parquet')
GROUP BY department
HAVING AVG(salary) > 60000
""").df())


# ==================================================
# Bonus Task 5
# Salary Category
# ==================================================

print("\nSalary Category")

print(duckdb.sql("""
SELECT
name,
salary,
CASE
WHEN salary >= 80000 THEN 'High'
WHEN salary >= 60000 THEN 'Medium'
ELSE 'Low'
END AS salary_category
FROM read_parquet('employees.parquet')
""").df())

print("\nAssignment Completed Successfully")