# AI-Data-Analyst
This project is a local AI-powered data analyst built with Python and Streamlit that enables users to upload any CSV or Excel dataset and interact with their data using natural language queries.
# 📊 Fine-Tuning Mistral-7B for Text-to-SQL Generation

This project demonstrates how to fine-tune a powerful instruction-tuned language model for translating natural language questions into SQL queries using **efficient training methods** like QLoRA and LoRA, all on limited hardware (Google Collab/Kaggle). The model is trained on the **Spider** dataset, a benchmark for complex, cross-domain text-to-SQL tasks.

---

## 🎯 Project Objective

Enable non-technical users to query structured databases using natural language by fine-tuning a large language model (LLM) to generate SQL queries.

### ✅ Use Case: Natural Language to SQL Query Generation
- **Automated Query Generation**: e.g., “Show me sales in Q1.”
- **Improved Accessibility**: No need to know SQL.
- **Reduces Engineering Bottlenecks**: Saves developer time on writing routine queries.

---

## 🛠️ Techniques Employed

### 1️⃣ Model: [`Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- Causal language model fine-tuned to follow instructions.
- Strong performance with fewer parameters than LLaMA 2.
- Suitable for generation tasks like SQL output.

## 🛠️ Tech Stack

### 🔍 Inference Agent (`agent.py`)
- **Model Inference**: [Ollama](https://ollama.com) (Locally hosted LLM serving)
- **LLM Model**: `mistral` (Mistral-7B via Ollama)
- **API Integration**: `requests` (Python HTTP library)

---

### 🌐 Web Application (`main.py`)
- **Frontend Framework**: `Streamlit` (for UI and interaction)
- **Data Handling**:
  - `pandas` (data processing and cleaning)
  - `duckdb` (SQL engine for querying uploaded data)
  - `csv`, `tempfile` (file handling utilities)
- **File Upload Support**: `.csv`, `.xlsx`

---

### 💾 File Handling & Preprocessing
- **Data Preprocessing**:
  - Type inference and conversion (object → numeric/date)
  - Missing value handling (`NA`, `N/A`, `missing`)
  - Schema extraction for prompt generation

---

### 🧪 Model Output
- **Prompt Template**: SQL generation based on table schema and user question
- **Query Execution**: Runs generated SQL on uploaded dataset using DuckDB
- **Output Display**: Interactive results shown via Streamlit table

---
# Fine Tuned Model

- **Trained dataset** within train.py

## 📚 Dataset: [Spider](https://yale-lily.github.io/spider)

- Complex, multi-domain SQL queries paired with natural language.
- Includes corresponding database schemas for real-world applicability.

---
