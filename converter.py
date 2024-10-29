import json

# Carga y convierte el archivo JSONL
def convertir_jsonl(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            messages = data.get("input", "")
            output = data.get("output", "")

            # Crear el nuevo formato de estructura de mensajes
            messages_formato = [
                {"role": "system", "content": "You are a helpful assistant and you're name is Acolo"},
                {"role": "user", "content": messages},
                {"role": "assistant", "content": output}
            ]

            # Generar la nueva estructura para cada entrada
            nueva_estructura = {"messages": messages_formato}

            # Guardar en el archivo de salida en el formato JSONL
            outfile.write(json.dumps(nueva_estructura) + '\n')

# Ejemplo de uso
convertir_jsonl('./dataset.jsonl', 'train.jsonl')