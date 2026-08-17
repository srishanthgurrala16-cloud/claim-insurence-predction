import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CREDITFAIR - DATASET EXPLORATION AND ANALYSIS
# ============================================================


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("1. LOAD DATASET")
print("=" * 70)

file_path = "/Users/padaltiruvinayak/Desktop/Credit ML/Dataset/creditfair_dataset.csv"

try:

    df = pd.read_csv(file_path, low_memory=False)

    print("Dataset loaded successfully!")

    # --------------------------------------------------------
    # Dataset shape
    # --------------------------------------------------------

    print("\n2. Number of Rows and Columns:")
    print("-----------------------------------")
    print("Rows    :", df.shape[0])
    print("Columns :", df.shape[1])


    # --------------------------------------------------------
    # Column names
    # --------------------------------------------------------

    print("\n3. Column Names:")
    print("-----------------------------------")

    for i, column in enumerate(df.columns, start=1):
        print(f"{i:2}. {column}")


    # ========================================================
    # 2. DATASET VIEW
    # ========================================================

    print("\n" + "=" * 70)
    print("4. FIRST 10 RECORDS")
    print("=" * 70)

    print(df.head(10))


    print("\n" + "=" * 70)
    print("5. LAST 10 RECORDS")
    print("=" * 70)

    print(df.tail(10))


    # ========================================================
    # 3. DATA TYPES
    # ========================================================

    print("\n" + "=" * 70)
    print("6. DATA TYPES")
    print("=" * 70)

    print(df.dtypes)


    # ========================================================
    # 4. COLUMN + DATA TYPE
    # ========================================================

    print("\n" + "=" * 70)
    print("7. COLUMN NAMES WITH DATA TYPES")
    print("=" * 70)

    print("\nColumn Name                 Data Type")
    print("-" * 45)

    for column in df.columns:
        print(f"{column:<28} {df[column].dtype}")


    # ========================================================
    # 5. DATASET INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("8. DATASET INFORMATION")
    print("=" * 70)

    df.info()


    # ========================================================
    # 6. NUMERICAL COLUMNS
    # ========================================================

    print("\n" + "=" * 70)
    print("9. NUMERICAL COLUMNS")
    print("=" * 70)

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numeric_columns:
        print(column)


    # ========================================================
    # 7. CATEGORICAL COLUMNS
    # ========================================================

    print("\n" + "=" * 70)
    print("10. CATEGORICAL COLUMNS")
    print("=" * 70)

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        print(column)


    # ========================================================
    # 8. MISSING VALUES
    # ========================================================

    print("\n" + "=" * 70)
    print("11. MISSING VALUES IN EACH COLUMN")
    print("=" * 70)

    missing_values = df.isnull().sum()

    print(missing_values)


    # --------------------------------------------------------
    # Total missing values
    # --------------------------------------------------------

    total_missing = df.isnull().sum().sum()

    print("\nTotal Missing Values:", total_missing)


    # ========================================================
    # 9. DUPLICATE RECORDS
    # ========================================================

    print("\n" + "=" * 70)
    print("12. DUPLICATE RECORDS")
    print("=" * 70)

    duplicate_count = df.duplicated().sum()

    print("Number of Duplicate Records:", duplicate_count)


    # ========================================================
    # 10. STATISTICAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("13. STATISTICAL SUMMARY")
    print("=" * 70)

    print(df.describe())


    # ========================================================
    # 11. LOAN STATUS DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("14. LOAN STATUS DISTRIBUTION")
    print("=" * 70)

    print(df["loan_status"].value_counts())


    # ========================================================
    # 12. LOAN STATUS PERCENTAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("15. LOAN STATUS PERCENTAGE")
    print("=" * 70)

    print(
        df["loan_status"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


    # ========================================================
    # 13. HISTOGRAM - LOAN AMOUNT
    # ========================================================

    print("\nDisplaying Loan Amount Histogram...")

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["loan_amnt"].dropna(),
        bins=20,
        edgecolor="black"
    )

    plt.title("Distribution of Loan Amount")
    plt.xlabel("Loan Amount")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()


    # ========================================================
    # 14. HISTOGRAM - ANNUAL INCOME
    # ========================================================

    print("Displaying Annual Income Histogram...")

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["annual_inc"].dropna(),
        bins=20,
        edgecolor="black"
    )

    plt.title("Distribution of Annual Income")
    plt.xlabel("Annual Income")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()


    # ========================================================
    # 15. LOAN STATUS COUNT PLOT
    # ========================================================

    print("Displaying Loan Status Distribution...")

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="loan_status"
    )

    plt.title("Loan Status Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Number of Loans")

    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.show()


    # ========================================================
    # 16. CORRELATION HEATMAP
    # ========================================================

    print("Displaying Correlation Heatmap...")

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()


    # ========================================================
    # COMPLETED
    # ========================================================

    print("\n" + "=" * 70)
    print("CREDITFAIR DATASET ANALYSIS COMPLETED")
    print("=" * 70)


except FileNotFoundError:

    print("\nERROR: Dataset file was not found.")
    print("Check the file path:")
    print(file_path)


except Exception as e:

    print("\nERROR:", e)