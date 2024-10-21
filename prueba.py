import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast

# Cargar el tokenizer con el nuevo comportamiento (legacy=False)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "meta-llama/Llama-3.2-1B", legacy=False
)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

checkpoint = torch.load(
    r"./results/checkpoint-3/rng_state.pth",
    map_location="cpu",
)
print(checkpoint.keys())

""" model.load_state_dict(
    checkpoint["model_state_dict"]
) """  # Asegúrate de que esta clave exista


# Cargar el modelo guardado en .pth
# model.load_state_dict(torch.load("./rng_state.pth"))
model.eval()  # Modo de evaluación


# Función para generar respuestas
def generar_respuesta(pregunta):
    inputs = tokenizer.encode(pregunta, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta


# Probar con una pregunta sobre finanzas
pregunta = "¿Qué es el Servicio de Administración Tributaria?"
respuesta = generar_respuesta(pregunta)
print(f"\nPregunta: {pregunta} \n")
print(f"\nRespuesta: {respuesta}")