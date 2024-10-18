import mlflow
import mlflow.pytorch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Configurar el URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://ec2-54-162-123-183.compute-1.amazonaws.com:5000")

# Ruta local donde tienes los archivos del modelo y el tokenizador
ruta_local_modelo = r"C:\Users\claud\Llama-3.2-1B"  # Usar 'r' para evitar problemas con rutas en Windows

# Cargar el tokenizador y el modelo de manera local
tokenizer = AutoTokenizer.from_pretrained(ruta_local_modelo)
model = AutoModelForCausalLM.from_pretrained(ruta_local_modelo, torch_dtype=torch.float16, device_map="auto")

# Crear pipeline para generación de texto usando el modelo local
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    torch_dtype=torch.float16,  # Puedes usar float16 o bfloat16 si tu hardware lo permite
    device_map="auto"           # Distribuye automáticamente a GPUs si están disponibles
)

# Iniciar una ejecución de MLflow
with mlflow.start_run(run_name="LLaMA_text_generation_experiment"):

    # Registrar parámetros del experimento
    mlflow.log_param("model_path", ruta_local_modelo)
    mlflow.log_param("torch_dtype", "float16")
    mlflow.log_param("device_map", "auto")

    # Prueba de generación de texto
    prompt = "The key to life is"
    max_length = 50
    num_return_sequences = 1
    output = pipe(prompt, max_length=max_length, num_return_sequences=num_return_sequences, do_sample=True)
    generated_text = output[0]['generated_text']
    
    # Registrar métrica: longitud del texto generado
    mlflow.log_metric("generated_text_length", len(generated_text))

    # Imprimir el texto generado en la terminal
    print("Texto generado:")
    print(generated_text)

    # Guardar el modelo completo en MLflow
    mlflow.pytorch.log_model(model, "llama_text_generation_model")

# Terminar la ejecución
print("Modelo y parámetros registrados con éxito en MLflow.")
