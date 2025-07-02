import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import duckdb
from agent import generate_sql_from_question  # Import the agent function

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        for col in df.columns:
            if 'date' in col.lower():
                df[col] == pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

st.title("📊 Local AI Data Analyst Agent")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if df is not None:
        st.write("Uploaded Data:")
        st.dataframe(df)
        st.write("Columns:", columns)

        # Connect to DuckDB and load the CSV data
        con = duckdb.connect()
        con.execute(f"CREATE OR REPLACE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{temp_path}')")

        user_query = st.text_area("Ask a question about your data:")

        if st.button("Submit"):
            with st.spinner("Generating SQL..."):
                # Include table name in the prompt explicitly
                table_name = "uploaded_data"
                table_desc = f"Table name: {table_name}\n" + "\n".join(
                    [f"- {col}: {str(df[col].dtype)}" for col in df.columns]
                )

                generated_sql = generate_sql_from_question(user_query, table_desc)

            st.markdown("**Generated SQL:**")
            st.code(generated_sql, language="sql")

            try:
                result_df = con.execute(generated_sql).fetchdf()
                st.success("Query executed successfully!")
                st.dataframe(result_df)
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")
