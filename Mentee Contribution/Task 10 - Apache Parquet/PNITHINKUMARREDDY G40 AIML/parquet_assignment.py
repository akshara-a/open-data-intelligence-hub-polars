import pandas as pd


def create_employee_dataframe():
    """Create and return an employee DataFrame."""

    data = {
        "employee_id": [1, 2, 3, 4, 5],
        "name": ["Asha", "Rahul", "Neha", "Vikram", "Priya"],
        "department": ["IT", "HR", "IT", "Finance", "HR"],
        "salary": [60000, 45000, 70000, 55000, 48000]
    }

    df = pd.DataFrame(data)

    # Extra Features
    df["bonus"] = df["salary"] * 0.10
    df["total_salary"] = df["salary"] + df["bonus"]

    return df


def main():

    # Task 1: Create Employee DataFrame
    employees_df = create_employee_dataframe()

    print("=" * 60)
    print("ALL EMPLOYEE RECORDS")
    print("=" * 60)
    print(employees_df)

    # Task 2: Save DataFrame as Parquet
    employees_df.to_parquet(
        "employees.parquet",
        engine="pyarrow",
        index=False
    )

    print("\nemployees.parquet created successfully.")

    # Task 3: Read Parquet File
    loaded_df = pd.read_parquet(
        "employees.parquet",
        engine="pyarrow"
    )

    print("\n" + "=" * 60)
    print("DATA READ FROM EMPLOYEES.PARQUET")
    print("=" * 60)
    print(loaded_df)

    # Task 4.1: Employees with salary > 50000
    high_salary_df = loaded_df[loaded_df["salary"] > 50000]

    print("\n" + "=" * 60)
    print("EMPLOYEES WITH SALARY GREATER THAN 50000")
    print("=" * 60)
    print(high_salary_df)

    # Task 4.2: Average Salary
    average_salary = loaded_df["salary"].mean()

    print("\n" + "=" * 60)
    print("AVERAGE EMPLOYEE SALARY")
    print("=" * 60)
    print(f"Average Salary: ₹{average_salary:,.2f}")

    # Task 4.3: Department Count
    department_counts = loaded_df["department"].value_counts()

    print("\n" + "=" * 60)
    print("NUMBER OF EMPLOYEES IN EACH DEPARTMENT")
    print("=" * 60)
    print(department_counts)

    # Task 5: Save High Salary Employees
    high_salary_df.to_parquet(
        "high_salary_employees.parquet",
        engine="pyarrow",
        index=False
    )

    print("\nhigh_salary_employees.parquet created successfully.")

    # Bonus Task: Read only Name and Salary columns
    selected_columns_df = pd.read_parquet(
        "employees.parquet",
        columns=["name", "salary"],
        engine="pyarrow"
    )

    print("\n" + "=" * 60)
    print("BONUS TASK: NAME AND SALARY COLUMNS")
    print("=" * 60)
    print(selected_columns_df)

    # ---------------- EXTRA FEATURES ----------------

    # Highest Paid Employee
    highest_paid = loaded_df.loc[loaded_df["salary"].idxmax()]

    print("\n" + "=" * 60)
    print("HIGHEST PAID EMPLOYEE")
    print("=" * 60)
    print(highest_paid)

    # Lowest Paid Employee
    lowest_paid = loaded_df.loc[loaded_df["salary"].idxmin()]

    print("\n" + "=" * 60)
    print("LOWEST PAID EMPLOYEE")
    print("=" * 60)
    print(lowest_paid)

    # Total Salary Paid
    total_salary = loaded_df["salary"].sum()

    print("\n" + "=" * 60)
    print("TOTAL SALARY PAID")
    print("=" * 60)
    print(f"₹{total_salary:,.2f}")

    # Average Bonus
    average_bonus = loaded_df["bonus"].mean()

    print("\n" + "=" * 60)
    print("AVERAGE BONUS")
    print("=" * 60)
    print(f"₹{average_bonus:,.2f}")

    # Department-wise Salary
    department_salary = loaded_df.groupby("department")["salary"].sum()

    print("\n" + "=" * 60)
    print("DEPARTMENT-WISE TOTAL SALARY")
    print("=" * 60)
    print(department_salary)

    print("\nTask 10 completed successfully!")


if __name__ == "__main__":
    main()