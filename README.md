# Open Model Tone Fine-tuning

LoRA fine-tuning открытой **русскоязычной** модели под **tone of voice** бизнес-коммуникации: спокойно, понятно, на практике — для малого бизнеса, клиентского сервиса и AI-автоматизации.

## Бизнес-задача

Малый бизнес и команды поддержки каждый день отвечают на повторяющиеся вопросы: цены, сроки, запись, задержки, возражения, негативные отзывы. Клиенту важны не только факты, но и **единый спокойный тон**: без давления, без жаргона, с понятным следующим шагом.

**Задача кейса:** показать полный цикл — от датасета до сравнения ответов — как через LoRA сместить компактную открытую модель к стилю AI-ассистента для бизнес-коммуникации, не претендуя на production-уровень качества.

## Что делает проект

- **`fine_tuning/train.py`** — обучение LoRA-адаптера; примеры в формате `### Instruction` / `### Response`.
- **`data/business_tone_dataset.json`** — основной датасет, 100 пар `instruction` / `output` на русском.
- **`inference/generate_once.py`** — **основной демонстрационный запуск**: один prompt в том же формате, что при обучении.
- **`inference/chat.py`** — интерактивный чат (`User:` / `Assistant:`); для сдачи не обязателен.

Базовая модель для основного сценария: **`ai-forever/rugpt3small_based_on_gpt2`** (прежний идентификатор `sberbank-ai/rugpt3small_based_on_gpt2` на Hugging Face перенаправляется на этот; в командах ниже используется актуальный). Локальная копия модели лежит в `models/<id с «/» → «_»>`, поэтому при смене идентификатора модель скачивается заново.

Среда: **Windows / PowerShell**, **Python 3.12**, **CPU**.  
В git не попадают: скачанные модели (`models/`), LoRA-веса, checkpoints, `.venv` — см. `.gitignore`.

## Структура проекта

```text
open-model-tone-finetuning/
├── data/
│   └── business_tone_dataset.json
├── fine_tuning/
│   └── train.py
├── inference/
│   ├── generate_once.py
│   ├── chat.py
│   └── preflight.py        # общие проверки перед загрузкой модели
├── tests/                  # лёгкие тесты (без скачивания моделей)
├── requirements.txt        # основной CPU-сценарий
├── requirements-dev.txt    # + pytest
├── requirements-gpu.txt    # необязательно: + bitsandbytes для --use_4bit на GPU
├── .gitignore
└── README.md
```

После запуска локально появляются каталоги вне git (см. `.gitignore`): `models/`, `.venv/`, `fine_tuning/tmp_cpu_test/`, `fine_tuning/lora_business_tone_cpu_3ep/`.

## Стек технологий

| Технология | Роль |
|------------|------|
| Python 3.12 | Запуск на CPU, Windows |
| PyTorch | Обучение и генерация |
| Hugging Face Transformers | Базовая модель и токенизатор |
| PEFT (LoRA) | Дообучение адаптера без полного fine-tune |
| datasets | Загрузка JSON-датасета |
| accelerate | Обучение на CPU / GPU |
| pytest | Только для тестов (`requirements-dev.txt`) |
| bitsandbytes | Необязательно: 4-bit на GPU (`requirements-gpu.txt`), для CPU-сценария не нужен |

## Датасет

**Файл:** `data/business_tone_dataset.json`

| Параметр | Значение |
|----------|----------|
| Объём | 100 пар |
| Поля | `instruction`, `output` |
| Язык | русский |
| Домен | малый бизнес, поддержка, AI-автоматизация |

**Тон ответов в датасете:** спокойно; понятно; без техножаргона и агрессивных продаж; практическая польза; завершение следующим шагом; без неподтверждённых цифр и без юридических/медицинских/финансовых гарантий.

В `train.py` каждая пара превращается в текст:

```text
### Instruction:
{instruction}

### Response:
{output}
```

`train.py` строго проверяет датасет **до загрузки модели**. Принимаются:

- `.json` — список записей либо объект `{"data": [...]}`;
- `.jsonl` — одна запись на строку (пустые строки пропускаются).

Запись — JSON-объект одной из схем: `text`; `instruction` + `output`; `prompt` + `completion`; `input` + `output`. Обязательные поля должны быть непустыми строками. Некорректный JSON, неподдерживаемая запись или пустой датасет останавливают запуск с указанием записи (для JSONL — номера строки); ничего не пропускается молча.

## Установка

```powershell
cd "путь\к\open-model-tone-finetuning"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Проверка окружения:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Для этого кейса ожидается **`CUDA: False`** (обучение и inference на CPU).

`requirements.txt` содержит только то, что нужно для основного CPU-сценария (`bitsandbytes` в него не входит). Дополнительно:

```powershell
# тесты (добавляет pytest)
python -m pip install -r .\requirements-dev.txt

# необязательно, только GPU + train.py --use_4bit (добавляет bitsandbytes)
python -m pip install -r .\requirements-gpu.txt
```

Для GPU-режима нужна сборка PyTorch с поддержкой CUDA; на CPU флаг `--use_4bit` отключается с предупреждением. GPU-режим в рамках последней проверки не запускался.

Проверка (тесты, CLI `--help`, `pip check`) выполнялась в окружении: Python 3.12.10, torch 2.12.0, transformers 5.9.0, peft 0.19.1, datasets 4.8.5, accelerate 1.13.0, pytest 9.1.1. Нижние границы версий в `requirements.txt` — заявленный минимум, отдельно они не проверялись.

## Smoke-test на маленькой модели

Проверка пайплайна на `sshleifer/tiny-gpt2` без долгого ожидания:

```powershell
python .\fine_tuning\train.py --model_name "sshleifer/tiny-gpt2" --dataset_path ".\data\business_tone_dataset.json" --output_dir ".\fine_tuning\tmp_cpu_test" --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_length 64 --device cpu
```

## Основное обучение на CPU

Три эпохи, `max_length 128`. Адаптер сохраняется в `fine_tuning/lora_business_tone_cpu_3ep` (только локально):

```powershell
python .\fine_tuning\train.py --model_name "ai-forever/rugpt3small_based_on_gpt2" --dataset_path ".\data\business_tone_dataset.json" --output_dir ".\fine_tuning\lora_business_tone_cpu_3ep" --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_length 128 --device cpu --seed 42
```

На CPU обучение может идти долго; планируйте время или уменьшайте число эпох для экспериментов.

### Воспроизводимость и проверка параметров

- **`--seed`** (по умолчанию **42**) задаёт seed для инициализации LoRA-матриц, порядка данных и dropout (`set_seed` + `TrainingArguments(seed=...)`). Цель — воспроизводимость на CPU при одинаковых версиях библиотек; на GPU полная детерминированность не гарантируется.
- Числовые параметры (`--num_train_epochs`, `--per_device_train_batch_size`, `--learning_rate`, `--lora_dropout`, `--seed` и др.) проверяются **до** загрузки данных и модели; при некорректном значении скрипт завершается с кодом 2 и сообщением об ошибке.
- Число шагов в логе считается как в `Trainer` (неполные батчи учитываются), поэтому для маленьких датасетов оно не показывается как 0.
- Модели загружаются без `trust_remote_code`: поддерживаемой GPT-2-модели сторонний код не нужен.

## Демонстрационный запуск одного запроса

Используйте **`inference/generate_once.py`**: промпт строится как при обучении (`### Instruction` + `### Response`), поэтому результат LoRA сравним с baseline.

`chat.py` использует другой шаблон (`User:` / `Assistant:`) — для демонстрации fine-tuning он менее показателен.

### Baseline (без LoRA)

```powershell
python .\inference\generate_once.py --base_model "ai-forever/rugpt3small_based_on_gpt2" --prompt "Зачем малому бизнесу нужен чат-бот на сайте?" --max_new_tokens 90 --temperature 0.1 --top_p 0.7
```

### LoRA (после обучения)

```powershell
python .\inference\generate_once.py --base_model "ai-forever/rugpt3small_based_on_gpt2" --lora_model ".\fine_tuning\lora_business_tone_cpu_3ep" --prompt "Зачем малому бизнесу нужен чат-бот на сайте?" --max_new_tokens 90 --temperature 0.1 --top_p 0.7
```

Скрипт выводит в терминал: базовую модель, статус LoRA, вопрос и ответ.

Генерация использует сэмплирование, но `--seed` (по умолчанию `42`) делает повторные baseline/LoRA-запуски воспроизводимее при одинаковом окружении. Для другого варианта ответа передайте, например, `--seed 7`.

**Если `--lora_model` указан, адаптер обязан загрузиться.** Путь должен существовать и содержать PEFT LoRA-адаптер (`adapter_config.json` с `peft_type=LORA` и файл весов `adapter_model.safetensors` или `adapter_model.bin`). Иначе скрипт завершается с ненулевым кодом (2) и сообщением об ошибке — он **не** переходит молча на baseline. Строка `LoRA: <путь> (адаптер загружен и объединён ...)` выводится только после реальной загрузки адаптера. Без `--lora_model` работает baseline (`LoRA: не используется (baseline)`). Так же ведёт себя `chat.py`.

Локальная копия базовой модели считается полной, только если в `models/<id>` есть `config.json` **и** файл весов (`model.safetensors`, `pytorch_model.bin` или индекс шардов); неполная директория не используется, модель загружается с Hugging Face. Параметры генерации проверяются до загрузки модели: `--max_new_tokens > 0`, `--temperature > 0`, `0 < --top_p <= 1`.

## Сравнение baseline vs LoRA

Один и тот же `--prompt`, одинаковые `--max_new_tokens`, `--temperature`, `--top_p`:

| | Baseline | С LoRA |
|---|----------|--------|
| Тема | Часто уходит мимо вопроса, общий или несвязанный текст | Чаще ближе к бизнес-коммуникации из датасета |
| Формат | Слабо следует шаблону ответа ассистента | Ближе к `### Response` и спокойному тону |
| Вывод | Показывает исходную базу | Показывает эффект адаптера |

Кейс демонстрирует **pipeline** (датасет → LoRA → inference), а не финальное production-качество. Таблица выше — качественное описание, а не измерение: метрик и eval-набора в проекте нет, конкретные ответы зависят от параметров сэмплирования и seed. Для отчёта сделайте два скриншота терминала подряд с одной командой, отличающейся только `--lora_model`.

## Тесты

Лёгкие тесты не скачивают модели (Hugging Face Hub в тестах принудительно офлайн) и не запускают обучение:

```powershell
python -m pip install -r .\requirements-dev.txt
python -m pytest -q
```

Они проверяют разбор и валидацию датасета, строгую обработку `--lora_model`, определение полного локального кэша модели, проверку аргументов, расчёт числа шагов и формат промпта `generate_once.py` (`### Instruction` / `### Response`). Качество модели тесты не оценивают.

## Ограничения

- Это **учебный demo-case**, не production-модель.
- Используется **маленькая GPT-2-подобная** модель; потолок качества ограничен размером базы.
- **Fine-tuning улучшает стиль и формат**, но **не гарантирует фактическую точность** фактов и рекомендаций.
- Модель может **галлюцинировать**, **повторяться** или давать **неидеальные** ответы.
- Обучение и генерация на **CPU** занимают много времени.
- Для production нужны: **более сильная базовая модель**, **расширенный датасет**, **eval-набор** и **human review**.

## Что демонстрирует кейс для портфолио

- Сбор и оформление доменного датасета под tone of voice (`data/business_tone_dataset.json`).
- LoRA fine-tuning открытой русскоязычной модели на CPU.
- Согласование формата **обучения** и **inference** (`train.py` + `generate_once.py`).
- Разделение кода/данных в репозитории и тяжёлых артефактов в `.gitignore`.
- Честное сравнение **baseline vs LoRA** без завышенных обещаний.

## Скриншоты для сдачи

1. Установка зависимостей (`pip install -r requirements.txt` или проверка `torch`).
2. Smoke-test — `train.py` + `tiny-gpt2`, ход обучения / завершение.
3. Основное обучение — `rugpt3small`, сохранение в `lora_business_tone_cpu_3ep`.
4. **Baseline** — `generate_once.py` без `--lora_model`.
5. **LoRA** — тот же prompt с `--lora_model .\fine_tuning\lora_business_tone_cpu_3ep`.
6. Фрагмент `data/business_tone_dataset.json` в редакторе.
7. `git status` — в индексе нет `models/`, `.venv/`, LoRA-папки, `temp_lesson/`.
