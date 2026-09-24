"""
Небольшие проверки для inference-скриптов: выполняются ДО загрузки модели.
"""
import json
import math
import os

# Локальная копия модели считается полной, если есть config.json и хотя бы один
# файл весов (или индекс шардов). Список совпадает с has_complete_local_model в fine_tuning/train.py.
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def has_complete_local_model(cache_dir):
    """True, только если в cache_dir есть config.json и хотя бы один файл весов/индекс."""
    if not os.path.isfile(os.path.join(cache_dir, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(cache_dir, name)) for name in MODEL_WEIGHT_FILES)


def validate_lora_adapter(lora_model_path):
    """
    Проверяет, что путь указывает на правдоподобный PEFT LoRA-адаптер.
    При любой проблеме бросает ValueError: тихого отката к базовой модели быть не должно.
    """
    if not os.path.isdir(lora_model_path):
        raise ValueError(f"LoRA-адаптер не найден (ожидается директория): {lora_model_path}")

    config_path = os.path.join(lora_model_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"В {lora_model_path} нет adapter_config.json - это не PEFT-адаптер")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать {config_path}: {exc}") from exc

    if not isinstance(config, dict) or str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"{config_path}: ожидается PEFT-адаптер с peft_type=LORA")

    if not any(os.path.isfile(os.path.join(lora_model_path, name)) for name in ADAPTER_WEIGHT_FILES):
        raise ValueError(
            f"В {lora_model_path} нет весов адаптера ({' или '.join(ADAPTER_WEIGHT_FILES)})"
        )


def validate_generation_args(length_name, length, temperature, top_p=None, top_k=None):
    """
    Проверяет параметры генерации (сэмплирование в обоих скриптах всегда включено).
    length_name - имя CLI-аргумента длины (max_new_tokens или max_length) для сообщения об ошибке.
    top_p и top_k проверяются, только если переданы.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError(f"--{length_name} должен быть целым числом > 0, получено: {length!r}")
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"--temperature должен быть > 0 (сэмплирование включено), получено: {temperature!r}")
    if top_p is not None and not (0 < top_p <= 1):
        raise ValueError(f"--top_p должен быть в диапазоне (0, 1], получено: {top_p!r}")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
        raise ValueError(f"--top_k должен быть целым числом >= 0 (0 отключает top-k), получено: {top_k!r}")
