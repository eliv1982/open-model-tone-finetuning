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

Базовая модель для основного сценария: **`sberbank-ai/rugpt3small_based_on_gpt2`**.

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
│   └── chat.py
├── requirements.txt
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

## Smoke-test на маленькой модели

Проверка пайплайна на `sshleifer/tiny-gpt2` без долгого ожидания:

```powershell
python .\fine_tuning\train.py --model_name "sshleifer/tiny-gpt2" --dataset_path ".\data\business_tone_dataset.json" --output_dir ".\fine_tuning\tmp_cpu_test" --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_length 64 --device cpu
```

## Основное обучение на CPU

Три эпохи, `max_length 128`. Адаптер сохраняется в `fine_tuning/lora_business_tone_cpu_3ep` (только локально):

```powershell
python .\fine_tuning\train.py --model_name "sberbank-ai/rugpt3small_based_on_gpt2" --dataset_path ".\data\business_tone_dataset.json" --output_dir ".\fine_tuning\lora_business_tone_cpu_3ep" --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_length 128 --device cpu
```

На CPU обучение может идти долго; планируйте время или уменьшайте число эпох для экспериментов.

## Демонстрационный запуск одного запроса

Используйте **`inference/generate_once.py`**: промпт строится как при обучении (`### Instruction` + `### Response`), поэтому результат LoRA сравним с baseline.

`chat.py` использует другой шаблон (`User:` / `Assistant:`) — для демонстрации fine-tuning он менее показателен.

### Baseline (без LoRA)

```powershell
python .\inference\generate_once.py --base_model "sberbank-ai/rugpt3small_based_on_gpt2" --prompt "Зачем малому бизнесу нужен чат-бот на сайте?" --max_new_tokens 90 --temperature 0.1 --top_p 0.7
```

### LoRA (после обучения)

```powershell
python .\inference\generate_once.py --base_model "sberbank-ai/rugpt3small_based_on_gpt2" --lora_model ".\fine_tuning\lora_business_tone_cpu_3ep" --prompt "Зачем малому бизнесу нужен чат-бот на сайте?" --max_new_tokens 90 --temperature 0.1 --top_p 0.7
```

Скрипт выводит в терминал: базовую модель, путь к LoRA (если есть), вопрос и ответ.

## Сравнение baseline vs LoRA

Один и тот же `--prompt`, одинаковые `--max_new_tokens`, `--temperature`, `--top_p`:

| | Baseline | С LoRA |
|---|----------|--------|
| Тема | Часто уходит мимо вопроса, общий или несвязанный текст | Чаще ближе к бизнес-коммуникации из датасета |
| Формат | Слабо следует шаблону ответа ассистента | Ближе к `### Response` и спокойному тону |
| Вывод | Показывает исходную базу | Показывает эффект адаптера |

Кейс демонстрирует **pipeline** (датасет → LoRA → inference), а не финальное production-качество. Для отчёта сделайте два скриншота терминала подряд с одной командой, отличающейся только `--lora_model`.

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
