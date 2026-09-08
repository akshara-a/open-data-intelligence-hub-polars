# Task 12 - Apache Arrow Assignment

**Student:** G Rajesh
**Batch:** G40 AI/ML
**Assignment:** Apache Arrow using Python

## Objective

The objective of this assignment is to understand and perform data processing operations using **Apache Arrow** and **PyArrow**.

## Technologies Used

* Python
* PyArrow
* Pandas
* Apache Arrow
* Parquet

## Tasks Completed

1. Created an Apache Arrow Table.
2. Displayed the Arrow Table schema.
3. Inspected rows, columns, and column names.
4. Selected specific columns.
5. Filtered employees with salary greater than 50,000.
6. Filtered employees belonging to the IT department.
7. Performed salary calculations:

   * Average salary
   * Maximum salary
   * Minimum salary
   * Total salary
8. Added a 10% bonus column.
9. Converted Arrow Table to Pandas DataFrame.
10. Converted Pandas DataFrame back to Arrow Table.
11. Saved the Arrow Table as a Parquet file.
12. Read the Parquet file.
13. Saved the Arrow Table as an Arrow IPC file.
14. Read the Arrow IPC file.

## Bonus Tasks

* Filtered employees from Delhi.
* Filtered employees with salary between 50,000 and 65,000.
* Added an annual salary column.
* Saved IT employees as a separate Parquet file.
* Read only the name and salary columns from the Parquet file.
* Sorted employees by salary in descending order.

## Project Structure

```text
Task12-Apache-Arrow-Assignment/
│
├── README.md
│
├── data/
│   ├── employees.arrow
│   └── employees.parquet
│
├── src/
│   └── apache_arrow_assignment.py
│
└── outputs/
    └── it_employees.parquet
```

## Dataset

The assignment uses sample employee data containing:

* Employee ID
* Name
* Department
* Salary
* City

The dataset is created directly in the Python script and converted into an Apache Arrow Table.

## Output Files

### `data/employees.parquet`

Contains the employee Arrow Table stored in Parquet format.

### `data/employees.arrow`

Contains the employee data stored in Arrow IPC format.

### `outputs/it_employees.parquet`

Contains employees belonging to the IT department.

## How to Run

Install the required libraries:

```bash
pip install pyarrow pandas
```

Run the Python program from the main assignment folder:

```bash
python src/apache_arrow_assignment.py
```

## Result

The assignment successfully demonstrates Apache Arrow table creation, schema inspection, filtering, column selection, calculations, Pandas conversion, Parquet file handling, Arrow IPC file handling, and additional data-processing operations using PyArrow.

## Conclusion

This assignment provided practical experience with **Apache Arrow and PyArrow** for efficient data processing and storage. It also demonstrated interoperability between Apache Arrow, Pandas, and Parquet formats.
