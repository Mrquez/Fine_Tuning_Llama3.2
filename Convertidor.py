import csv
import json

# Archivo CSV con preguntas y respuestas
csv_file = "faq_adeudos_ready.csv"
jsonl_file = "dataset_faq_adeudos.jsonl"

# Lee el archivo CSV y convierte a JSONL
with open(csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    with open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        for row in reader:
            jsonl_object = {
                "input": row['pregunta'],  # Columna de preguntas
                "output": row['respuesta']  # Columna de respuestas
            }
            jsonlfile.write(json.dumps(jsonl_object) + '\n')
