# 📊 Day 4 – Pandas DataFrames

I explored how to work with **Pandas DataFrames**, which are powerful 2D tabular data structures in Python. Below is a summary of what was covered.

---

## 🧱 1. DataFrames Basics

- **Creating DataFrames** from dictionaries or lists.
- **Inspecting structure**: `head()`, `tail()`, `shape`, `info()`, `dtypes`.
- **Indexing & Selection**:
  - Using column names
  - `.loc[]` – label-based access
  - `.iloc[]` – position-based access
- **Index basics**:
  - Setting custom indexes
  - Resetting indexes

---

## 🔎 2. Filtering Data

Techniques to filter rows in a DataFrame:
- **Bracket notation** – `df[df['col'] > 5]`
- **`.isin()`** – Select rows where column values match a list.
- **`.between()`** – Filter within a numeric range.
- **`.str.contains()`** – Filter strings with pattern matching.
- **`.query()`** – SQL-like query strings for filtering.
- **`.filter()`** – Select specific columns or rows by label or regex.

---

## ⚙️ 3. Useful Methods

Various built-in methods that simplify data analysis:

- **`.apply()`** – Apply custom functions to rows/columns.
- **`.describe()`** – Summary stats for numeric data.
- **`.sort_values()`** – Sort by one or more columns.
- **`.corr()`** – Correlation matrix.
- **`.idxmin()` / `.idxmax()`** – Return index of min/max value.
- **`.value_counts()`** – Frequency of unique values.
- **`.replace()`** – Replace values in the DataFrame.
- **`.unique()` / `.nunique()`** – Get unique values or count of them.
- **`.map()`** – Element-wise transformation for a Series.
- **`.duplicated()` / `.drop_duplicates()`** – Detect or remove duplicates.

---





