import json
import os

def build_schema_map(tables_json_path):
    with open(tables_json_path, "r") as f:
        tables = json.load(f)

    schema_map = {}
    for db in tables:
        db_id = db["db_id"]
        schema_strs = []
        for table_idx, table_name in enumerate(db["table_names_original"]):
            cols = [col[1] for col in db["column_names_original"] if col[0] == table_idx and col[1] != "*"]
            schema_strs.append(f"{table_name}({', '.join(cols)})")
        schema_map[db_id] = "\n".join(schema_strs)
    return schema_map

def prepare_dataset(train_path, tables_path, output_path):
    schema_map = build_schema_map(tables_path)

    with open(train_path, "r") as f:
        spider_data = json.load(f)

    output_lines = []
    for item in spider_data:
        question = item["question"].strip()
        sql = item["query"].strip()
        db_id = item["db_id"]

        if db_id not in schema_map:
            continue

        schema = schema_map[db_id]

        prompt = f"""You are a professional data analyst. Given the table below, write an SQL query to answer the user's question.

Table Schema:
{schema}

User Question:
{question}

Return only the SQL query, no explanation."""

        output_lines.append({"prompt": prompt, "response": sql})

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for line in output_lines:
            f_out.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(output_lines)} examples to {output_path}")


# Run it
prepare_dataset(
    train_path="../spider/train_spider.json",
    tables_path="../spider/tables.json",
    output_path="mistral_spider_train.jsonl"
)
