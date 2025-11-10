import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Загружаем токен
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Подключаемся к API StarCoder
client = InferenceClient(
    model="bigcode/starcoder",
    token=HF_TOKEN
)

def generate_code(prompt: str) -> str:
    """
    Отправляет запрос к StarCoder и возвращает сгенерированный код.
    """
    response = client.text_generation(
        prompt,
        max_new_tokens=200,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    return response
