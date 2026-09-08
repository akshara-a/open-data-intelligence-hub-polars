import pandas as pd
import duckdb

# ==========================================================
# Task 1: Create Parquet File
# ==========================================================

data = {
    "employee_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": [
        "Asha",
        "Rahul",
        "Neha",
        "Vikram",
        "Priya",
        "Arjun",
        "Meera",
        "Karan"
    ],
    "department": [
        "IT",
        "HR",
        "IT",
        "Finance",
        "HR",
        "Finance",
        "IT",
        "Sales"
    ],
    "salary": [
        60000,
        45000,
        70000,
        55000,
        48000,
        65000,
        75000,
        50000
    ],
    "city": [
        "Delhi",
        "Mumbai",
        "Bengaluru",
        "Delhi",
        "Mumbai",
        "Chennai",
        "Bengaluru",
        "Delhi"
    ]
}

df = pd.DataFrame(data)

df.to_parquet("employees.parquet", index=False)

print("=" * 60)
print("Task 1: Parquet File Created")
print("=" * 60)
print(df)


# ==========================================================
# Task 2: Read Parquet Using DuckDB
# ==========================================================

print("\n")
print("=" * 60)
print("Task 2: Read Parquet")
print("=" * 60)

result = duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
""").df()

print(result)


# ==========================================================
# Task 3: Filter Employee Records
# ==========================================================

print("\n")
print("=" * 60)
print("Task 3A: Salary > 50000")
print("=" * 60)

high_salary = duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
WHERE salary > 50000
""").df()

print(high_salary)


print("\n")
print("=" * 60)
print("Task 3B: IT Department")
print("=" * 60)

it_emp = duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
WHERE department='IT'
""").df()

print(it_emp)


print("\n")
print("=" * 60)
print("Task 3C: Employees from Delhi")
print("=" * 60)

delhi_emp = duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
WHERE city='Delhi'
""").df()

print(delhi_emp)


print("\n")
print("=" * 60)
print("Task 3D: IT Department Salary > 65000")
print("=" * 60)

it_high = duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
WHERE department='IT'
AND salary>65000
""").df()

print(it_high)


# ==========================================================
# Task 4: Select Specific Columns
# ==========================================================

print("\n")
print("=" * 60)
print("Task 4")
print("=" * 60)

selected = duckdb.sql("""
SELECT
    name,
    department,
    salary
FROM read_parquet('employees.parquet')
ORDER BY salary DESC
""").df()

print(selected)


# ==========================================================
# Task 5: Aggregations
# ==========================================================

print("\n")
print("=" * 60)
print("Task 5")
print("=" * 60)

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


# ==========================================================
# Task 6: Group By Department
# ==========================================================

print("\n")
print("=" * 60)
print("Task 6")
print("=" * 60)

grouped = duckdb.sql("""
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

print(grouped)


# ==========================================================
# Task 7: Create DuckDB Database
# ==========================================================

print("\n")
print("=" * 60)
print("Task 7")
print("=" * 60)

connection = duckdb.connect("company.duckdb")

connection.execute("""
CREATE OR REPLACE TABLE employees AS
SELECT *
FROM read_parquet('employees.parquet')
""")

table_data = connection.execute("""
SELECT *
FROM employees
""").df()

print(table_data)

connection.close()


# ==========================================================
# Task 8: Export High Salary Employees
# ==========================================================

print("\n")
print("=" * 60)
print("Task 8")
print("=" * 60)

duckdb.sql("""
COPY(
SELECT *
FROM read_parquet('employees.parquet')
WHERE salary>50000
)
TO 'high_salary_employees.parquet'
(FORMAT PARQUET)
""")

print("high_salary_employees.parquet created successfully.")


# ==========================================================
# Task 9: Verify Exported File
# ==========================================================

print("\n")
print("=" * 60)
print("Task 9")
print("=" * 60)

verify = duckdb.sql("""
SELECT *
FROM read_parquet('high_salary_employees.parquet')
""").df()

print(verify)


# ==========================================================
# Bonus Task 1
# Second Highest Salary
# ==========================================================

print("\n")
print("=" * 60)
print("Bonus 1: Second Highest Salary")
print("=" * 60)

second = duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
ORDER BY salary DESC
LIMIT 1 OFFSET 1
""").df()

print(second)


# ==========================================================
# Bonus Task 2
# Top 3 Highest Paid
# ==========================================================

print("\n")
print("=" * 60)
print("Bonus 2: Top 3 Highest Paid")
print("=" * 60)

top3 = duckdb.sql("""
SELECT *
FROM read_parquet('employees.parquet')
ORDER BY salary DESC
LIMIT 3
""").df()

print(top3)


# ==========================================================
# Bonus Task 3
# Average Salary by City
# ==========================================================

print("\n")
print("=" * 60)
print("Bonus 3")
print("=" * 60)

city_avg = duckdb.sql("""
SELECT
    city,
    AVG(salary) AS average_salary
FROM read_parquet('employees.parquet')
GROUP BY city
ORDER BY average_salary DESC
""").df()

print(city_avg)


# ==========================================================
# Bonus Task 4
# Departments Avg Salary > 55000
# ==========================================================

print("\n")
print("=" * 60)
print("Bonus 4")
print("=" * 60)

dept_avg = duckdb.sql("""
SELECT
    department,
    AVG(salary) AS average_salary
FROM read_parquet('employees.parquet')
GROUP BY department
HAVING AVG(salary)>55000
""").df()

print(dept_avg)


# ==========================================================
# Bonus Task 5
# Salary Category
# ==========================================================

print("\n")
print("=" * 60)
print("Bonus 5")
print("=" * 60)

salary_category = duckdb.sql("""
SELECT
    name,
    salary,
    CASE
        WHEN salary>=65000 THEN 'High'
        WHEN salary>=50000 THEN 'Medium'
        ELSE 'Low'
    END AS salary_category
FROM read_parquet('employees.parquet')
""").df()

print(salary_category)

print("\nAssignment Completed Successfully.")