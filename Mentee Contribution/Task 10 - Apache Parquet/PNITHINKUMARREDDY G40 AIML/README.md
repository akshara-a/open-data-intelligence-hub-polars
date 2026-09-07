# Task 10 – Apache Parquet

## Submitted By

**Name:** P. Shadik Khan  
**Batch:** G40 AI ML

---

## Objective

The objective of this project is to understand how to use Apache Parquet files with Python, Pandas, and PyArrow. This project demonstrates creating, storing, reading, filtering, and analyzing employee data using the Parquet file format.

---

## Technologies Used

- Python
- Pandas
- PyArrow
- Apache Parquet

---

## Files Included

- `parquet_assignment.py` – Main Python program
- `employees.parquet` – Stores all employee records
- `high_salary_employees.parquet` – Stores employees whose salary is greater than ₹50,000
- `requirements.txt` – Required Python libraries
- `README.md` – Project documentation

---

## Employee Dataset

| Employee ID | Name   | Department | Salary |
|-------------|--------|------------|--------|
| 1 | Asha   | IT       | 60000 |
| 2 | Rahul  | HR       | 45000 |
| 3 | Neha   | IT       | 70000 |
| 4 | Vikram | Finance  | 55000 |
| 5 | Priya  | HR       | 48000 |

---

## Tasks Completed

### Task 1
Created a Pandas DataFrame with employee information.

### Task 2
Saved the employee data as `employees.parquet`.

### Task 3
Read the Parquet file successfully.

### Task 4
Performed employee data analysis.

- Displayed employees earning more than ₹50,000.
- Calculated average salary.
- Counted employees in each department.

### Task 5
Created `high_salary_employees.parquet`.

### Bonus Task
Read only the **Name** and **Salary** columns from the Parquet file.

---

## Additional Features

- Added Bonus (10% of Salary)
- Added Total Salary (Salary + Bonus)
- Displayed Highest Paid Employee
- Displayed Lowest Paid Employee
- Calculated Total Salary Paid
- Calculated Average Bonus
- Displayed Department-wise Total Salary

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Program

```bash
python parquet_assignment.py
```

---

## Expected Output

- Employee records displayed
- Parquet file created successfully
- Parquet file read successfully
- High salary employees displayed
- Average salary calculated
- Department-wise employee count displayed
- Highest paid employee displayed
- Lowest paid employee displayed
- Department-wise salary displayed
- Bonus task completed successfully

---

## Conclusion

This project demonstrates efficient storage and analysis of structured data using the Apache Parquet format with Python. It covers data creation, reading, filtering, aggregation, and basic employee analytics while showcasing the advantages of Parquet for data engineering workflows.

---

**Submitted by**

**P. Shadik Khan**  
**G40 AI ML**