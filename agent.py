import requests 
 
OLLAMA_MODEL="mistral"

def generate_sql_from_question(question, table_description):
    prompt=f""" 
    You are a professional data analyst. Given the table below, write an SQL query to answer the 
    user's question.

    Table Schema:
    {table_description}

    User Question:
    {question}

    Return only the SQL query, no explanation.
    """ 

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":OLLAMA_MODEL,
                "prompt":prompt,
                "stream":False}
        )

        result = response.json()

        return result.get("response","").strip()

    except Exception as e:
        return f"Error generating SQL:{e}"