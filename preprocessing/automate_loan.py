import pandas as pd
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def process_data(input_path: str, output_dir: str):
    print(f"Membaca data dari {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Buang ID
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # 2. Handling Missing Values (Median untuk angka, Modus untuk teks)
    num_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]
    cat_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    # pastikan kolomnya ada (biar nggak KeyError kalau dataset beda)
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Encoding Target
    if "Loan_Status" not in df.columns:
        raise ValueError("Kolom 'Loan_Status' tidak ditemukan di dataset.")

    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    # kalau ada label aneh selain Y/N (hasilnya NaN), stop biar ketahuan
    if df["Loan_Status"].isna().any():
        bad_rows = df[df["Loan_Status"].isna()].head(5)
        raise ValueError(
            "Ada nilai Loan_Status selain 'Y' atau 'N'. Contoh baris bermasalah:\n"
            + bad_rows.to_string(index=False)
        )

    # 4. One-Hot Encoding fitur kategorikal
    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)

    # 5. Scaling fitur numerik tertentu
    scaler = StandardScaler()
    scale_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    scale_cols = [c for c in scale_cols if c in df.columns]
    if scale_cols:
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # 6. Split & Save
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Stratify penting karena klasifikasi (pastikan minimal 2 kelas ada)
    if y.nunique() < 2:
        raise ValueError("Target hanya punya 1 kelas. Stratify tidak bisa dilakukan.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)

    train_set = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_set = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train_path = os.path.join(output_dir, "train_clean.csv")
    test_path = os.path.join(output_dir, "test_clean.csv")

    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)

    print(f"Data bersih disimpan di folder: {output_dir}")
    print(f"Train: {train_path} | shape: {train_set.shape}")
    print(f"Test : {test_path}  | shape: {test_set.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path ke dataset raw CSV")
    parser.add_argument("--output", required=True, help="Folder output untuk train_clean.csv & test_clean.csv")
    args = parser.parse_args()

    process_data(args.input, args.output)
