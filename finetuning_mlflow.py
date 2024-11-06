import mlflow
import mlflow.transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Configurar el servidor remoto de MLflow
mlflow.set_tracking_uri("http://ec2-54-162-123-183.compute-1.amazonaws.com:5000")
mlflow.set_experiment("Fine-Tuning Llama 3.2")

# Cargar el modelo Llama 3.2 (1B parámetros) y el tokenizador
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Asignar el eos_token como pad_token
tokenizer.pad_token = tokenizer.eos_token

# Cargar el dataset en formato JSONL
dataset = load_dataset('json', data_files='dataset.jsonl')

# Tokenizar el dataset, incluyendo las etiquetas (labels)
def tokenize_function(examples):
    # Tokenizar input (pregunta) y output (respuesta)
    inputs = tokenizer(examples['input'], padding='max_length', truncation=True, max_length=128)
    labels = tokenizer(examples['output'], padding='max_length', truncation=True, max_length=128)
    
    # Añadir las etiquetas (labels) al conjunto de datos
    inputs['labels'] = labels['input_ids']
    
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Dividir el dataset en entrenamiento y validación
train_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1)

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results', 
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2
)

# Usar MLflow para el registro de experimentos
with mlflow.start_run():
    
    # Registrar hiperparámetros en MLflow
    mlflow.log_params({
        "batch_size": training_args.per_device_train_batch_size,
        "num_train_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
    })

    # Definir el Trainer
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset['train'],  # Conjunto de entrenamiento
        eval_dataset=train_dataset['test']     # Conjunto de validación
    )

    # (Opcional) Ver un ejemplo del dataset tokenizado
    print(train_dataset['train'][0])

    # Iniciar el entrenamiento
    trainer.train()

    # Registrar el modelo y las métricas en MLflow
    mlflow.transformers.log_model(trainer.model, "model")
    metrics = trainer.state.log_history[-1]  # Extraer las métricas del último paso
    mlflow.log_metrics(metrics)
