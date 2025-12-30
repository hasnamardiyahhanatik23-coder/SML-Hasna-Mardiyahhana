import pandas as pd
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def process_data(input_path, output_dir):
    print(f"Membaca data dari {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Buang ID
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    # 2. Handling Missing Values (Median untuk angka, Modus untuk teks)
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Encoding Target
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # 4. One-Hot Encoding Fitur
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 5. Scaling
    scaler = StandardScaler()
    scale_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # 6. Split & Save
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Stratify penting karena klasifikasi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Gabung kembali X dan y untuk disimpan sebagai CSV siap pakai
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    
    train_set.to_csv(os.path.join(output_dir, 'train_clean.csv'), index=False)
    test_set.to_csv(os.path.join(output_dir, 'test_clean.csv'), index=False)
    print(f"Data bersih disimpan di folder: {output_dir}")

if _name_ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    process_data(args.input, args.output)
