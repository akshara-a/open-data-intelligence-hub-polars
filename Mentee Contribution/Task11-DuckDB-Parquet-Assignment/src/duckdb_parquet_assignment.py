import pandas as pd
import duckdb
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATABASE_DIR = BASE_DIR / "database"

EMPLOYEES_FILE = DATA_DIR / "employees.parquet"
HIGH_SALARY_FILE = OUTPUTS_DIR / "high_salary_employees.parquet"
DATABASE_FILE = DATABASE_DIR / "company.duckdb"

# Create folders if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)


# ============================================================
# Task 1: Create the Parquet File
# ============================================================

data = {
    "employee_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": [
        "Asha", "Rahul", "Neha", "Vikram",
        "Priya", "Arjun", "Meera", "Karan"
    ],
    "department": [
        "IT", "HR", "IT", "Finance",
        "HR", "Finance", "IT", "Sales"
    ],
    "salary": [
        60000, 45000, 70000, 55000,
        48000, 65000, 75000, 50000
    ],
    "city": [
        "Delhi", "Mumbai", "Bengaluru", "Delhi",
        "Mumbai", "Chennai", "Bengaluru", "Delhi"
    ]
}

df = pd.DataFrame(data)

df.to_parquet(EMPLOYEES_FILE, index=False)

print("=" * 60)
print("Task 1: Parquet file created successfully.")
print("=" * 60)


# ============================================================
# Task 2: Read Parquet Using DuckDB
# ============================================================

result = duckdb.sql(f"""
    SELECT *
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
""").df()

print("\nTask 2: All Employee Records")
print(result)


# ============================================================
# Task 3: Filter Employee Records
# ============================================================

high_salary = duckdb.sql(f"""
    SELECT *
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    WHERE salary > 50000
""").df()

print("\nTask 3.1: Employees with Salary Greater Than 50000")
print(high_salary)


it_employees = duckdb.sql(f"""
    SELECT *
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    WHERE department = 'IT'
""").df()

print("\nTask 3.2: Employees from IT Department")
print(it_employees)


delhi_employees = duckdb.sql(f"""
    SELECT *
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    WHERE city = 'Delhi'
""").df()

print("\nTask 3.3: Employees from Delhi")
print(delhi_employees)


it_high_salary = duckdb.sql(f"""
    SELECT *
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    WHERE department = 'IT'
      AND salary > 65000
""").df()

print("\nTask 3.4: IT Employees with Salary Greater Than 65000")
print(it_high_salary)


# ============================================================
# Task 4: Select Specific Columns and Sort
# ============================================================

selected_columns = duckdb.sql(f"""
    SELECT name, department, salary
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    ORDER BY salary DESC
""").df()

print("\nTask 4: Name, Department and Salary Sorted by Salary")
print(selected_columns)


# ============================================================
# Task 5: Perform Aggregations
# ============================================================

summary = duckdb.sql(f"""
    SELECT
        COUNT(*) AS employee_count,
        AVG(salary) AS average_salary,
        MAX(salary) AS maximum_salary,
        MIN(salary) AS minimum_salary,
        SUM(salary) AS total_salary
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
""").df()

print("\nTask 5: Employee Salary Summary")
print(summary)


# ============================================================
# Task 6: Group Data by Department
# ============================================================

department_summary = duckdb.sql(f"""
    SELECT
        department,
        COUNT(*) AS employee_count,
        AVG(salary) AS average_salary,
        MAX(salary) AS highest_salary,
        SUM(salary) AS total_salary
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    GROUP BY department
    ORDER BY average_salary DESC
""").df()

print("\nTask 6: Department Salary Summary")
print(department_summary)


# ============================================================
# Task 7: Create a DuckDB Database
# ============================================================

connection = duckdb.connect(str(DATABASE_FILE))

connection.execute(f"""
    CREATE OR REPLACE TABLE employees AS
    SELECT *
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
""")

database_result = connection.execute("""
    SELECT *
    FROM employees
""").df()

print("\nTask 7: Employees Table from DuckDB Database")
print(database_result)

connection.close()

print("\ncompany.duckdb database created successfully.")


# ============================================================
# Task 8: Export High Salary Employees to Parquet
# ============================================================

duckdb.sql(f"""
    COPY (
        SELECT *
        FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
        WHERE salary > 50000
    )
    TO '{HIGH_SALARY_FILE.as_posix()}'
    (FORMAT PARQUET)
""")

print("\nTask 8: high_salary_employees.parquet created successfully.")


# ============================================================
# Task 9: Verify Exported Parquet File
# ============================================================

exported_result = duckdb.sql(f"""
    SELECT *
    FROM read_parquet('{HIGH_SALARY_FILE.as_posix()}')
""").df()

print("\nTask 9: High Salary Employees from Exported Parquet")
print(exported_result)


# ============================================================
# Bonus 1: Second-Highest Salary
# ============================================================

second_highest = duckdb.sql(f"""
    SELECT *
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    ORDER BY salary DESC
    LIMIT 1 OFFSET 1
""").df()

print("\nBonus 1: Second-Highest Salary Employee")
print(second_highest)


# ============================================================
# Bonus 2: Top Three Highest-Paid Employees
# ============================================================

top_three = duckdb.sql(f"""
    SELECT *
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    ORDER BY salary DESC
    LIMIT 3
""").df()

print("\nBonus 2: Top Three Highest-Paid Employees")
print(top_three)


# ============================================================
# Bonus 3: Average Salary for Each City
# ============================================================

city_salary = duckdb.sql(f"""
    SELECT
        city,
        AVG(salary) AS average_salary
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    GROUP BY city
    ORDER BY average_salary DESC
""").df()

print("\nBonus 3: Average Salary by City")
print(city_salary)


# ============================================================
# Bonus 4: Departments with Average Salary Greater Than 55000
# ============================================================

high_average_departments = duckdb.sql(f"""
    SELECT
        department,
        AVG(salary) AS average_salary
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    GROUP BY department
    HAVING AVG(salary) > 55000
    ORDER BY average_salary DESC
""").df()

print("\nBonus 4: Departments with Average Salary Greater Than 55000")
print(high_average_departments)


# ============================================================
# Bonus 5: Salary Category
# ============================================================

salary_categories = duckdb.sql(f"""
    SELECT
        name,
        salary,
        CASE
            WHEN salary >= 65000 THEN 'High'
            WHEN salary >= 50000 THEN 'Medium'
            ELSE 'Low'
        END AS salary_category
    FROM read_parquet('{EMPLOYEES_FILE.as_posix()}')
    ORDER BY salary DESC
""").df()

print("\nBonus 5: Salary Categories")
print(salary_categories)


print("\n" + "=" * 60)
print("TASK 11 COMPLETED SUCCESSFULLY!")
print("=" * 60)