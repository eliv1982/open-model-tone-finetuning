"""
Общие настройки тестов. Тесты не скачивают модели и не запускают обучение.
"""
import os
import sys

# Жёстко офлайн: любая случайная попытка обратиться к Hugging Face Hub упадёт, а не скачает модель.
# Переменные нужно выставить до импорта transformers/huggingface_hub (их импортируют сами скрипты).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Скрипты запускаются как `python fine_tuning\train.py` / `python inference\...`, а не как пакеты
for subdir in ("fine_tuning", "inference"):
    path = os.path.join(PROJECT_ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)
