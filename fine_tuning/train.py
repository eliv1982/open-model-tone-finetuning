"""
Скрипт для fine-tuning модели с использованием LoRA
"""
import os
import json
import math
import time
import sys
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import torch
from transformers import BitsAndBytesConfig

# Файлы весов, наличие которых (вместе с config.json) считается полной локальной копией модели.
# Список совпадает с inference/preflight.py (MODEL_WEIGHT_FILES).
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)

# Явно поддерживаемые схемы записей датасета: обязательные поля -> как собрать текст примера.
# Записи проверяются в этом порядке; побеждает первая схема, все поля которой есть в записи.
DATASET_SCHEMAS = {
    ("text",): lambda e: e["text"],
    ("instruction", "output"): lambda e: f"### Instruction:\n{e['instruction']}\n\n### Response:\n{e['output']}",
    ("prompt", "completion"): lambda e: f"{e['prompt']}\n\n{e['completion']}",
    ("input", "output"): lambda e: f"Input: {e['input']}\nOutput: {e['output']}",
}

class InvalidArguments(ValueError):
    """Некорректные аргументы запуска (проверяются до загрузки модели)."""

class CausalLMCollator:
    """Динамически дополняет батч и маскирует loss только на padding-позициях."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"].masked_fill_(batch["attention_mask"].eq(0), -100)
        return batch

def has_complete_local_model(cache_dir):
    """True, только если в cache_dir есть config.json и хотя бы один файл весов/индекс."""
    if not os.path.isfile(os.path.join(cache_dir, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(cache_dir, name)) for name in MODEL_WEIGHT_FILES)

def format_metric(value, fmt):
    """Безопасно форматирует метрику, даже если она пришла строкой."""
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)

def last_logged_loss(log_history):
    """Последний залогированный loss (в конце обучения последняя запись - сводка без 'loss')."""
    for entry in reversed(log_history or []):
        if 'loss' in entry:
            return entry['loss']
    return 'N/A'

def compute_training_steps(num_examples, batch_size, gradient_accumulation_steps, num_train_epochs):
    """
    Число шагов оптимизатора как в Trainer (один процесс, drop_last=False):
    неполный последний батч и неполная группа накопления градиента тоже дают шаг.
    Возвращает (шагов в эпохе, всего шагов).
    """
    batches_per_epoch = math.ceil(num_examples / batch_size)
    steps_per_epoch = max(math.ceil(batches_per_epoch / gradient_accumulation_steps), 1)
    return steps_per_epoch, math.ceil(num_train_epochs * steps_per_epoch)

def validate_train_args(
    *,
    num_train_epochs,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    learning_rate,
    max_length,
    lora_r,
    lora_alpha,
    lora_dropout,
    save_steps,
    logging_steps,
    warmup_steps,
    seed,
):
    """Проверяет числовые параметры обучения до загрузки модели. Бросает InvalidArguments."""
    def is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    for name, value in (
        ("num_train_epochs", num_train_epochs),
        ("learning_rate", learning_rate),
        ("lora_alpha", lora_alpha),
    ):
        if not (math.isfinite(value) and value > 0):
            raise InvalidArguments(f"--{name} должен быть числом > 0, получено: {value!r}")

    for name, value in (
        ("per_device_train_batch_size", per_device_train_batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("max_length", max_length),
        ("lora_r", lora_r),
        ("save_steps", save_steps),
        ("logging_steps", logging_steps),
    ):
        if not (is_int(value) and value > 0):
            raise InvalidArguments(f"--{name} должен быть целым числом > 0, получено: {value!r}")

    if not (is_int(warmup_steps) and warmup_steps >= 0):
        raise InvalidArguments(f"--warmup_steps должен быть целым числом >= 0, получено: {warmup_steps!r}")
    if not (math.isfinite(lora_dropout) and 0 <= lora_dropout < 1):
        raise InvalidArguments(f"--lora_dropout должен быть в диапазоне [0, 1), получено: {lora_dropout!r}")
    if not (is_int(seed) and 0 <= seed < 2**32):
        raise InvalidArguments(f"--seed должен быть целым числом от 0 до 2**32 - 1, получено: {seed!r}")

class DetailedLoggingCallback(TrainerCallback):
    """Callback для детального логирования процесса обучения"""

    def __init__(self, steps_per_epoch=None):
        self.start_time = None
        self.epoch_start_time = None
        self.steps_per_epoch = steps_per_epoch
        self.epoch_number = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Вызывается в начале обучения"""
        self.start_time = time.time()
        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        print(f"\n{'='*60}")
        print(f"ОБУЧЕНИЕ | Эпох: {args.num_train_epochs} | Батч: {effective_batch} | LR: {args.learning_rate}")
        print(f"{'='*60}\n")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Вызывается в начале каждой эпохи"""
        self.epoch_start_time = time.time()
        # state.epoch на начало эпохи ещё хранит число завершённых эпох (0.0 перед первой), поэтому свой счётчик
        self.epoch_number += 1
        steps = self.steps_per_epoch if self.steps_per_epoch else '?'
        print(f"\nЭпоха {self.epoch_number}/{args.num_train_epochs:g} | Шагов: {steps}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Вызывается при каждом логировании"""
        if logs is None:
            return
        
        step = state.global_step
        loss = logs.get('loss', 'N/A')
        lr = logs.get('learning_rate', 'N/A')
        loss_str = format_metric(loss, ".4f")
        lr_str = format_metric(lr, ".2e")
        
        # Компактный вывод
        if state.max_steps:
            progress = (step / state.max_steps) * 100
            print(f"Шаг {step}/{state.max_steps} ({progress:.1f}%) | Loss: {loss_str} | LR: {lr_str}", end='')
        else:
            print(f"Шаг {step} | Loss: {loss_str} | LR: {lr_str}", end='')
        
        # Память GPU (только если используется)
        if torch.cuda.is_available() and not getattr(args, "use_cpu", False):
            mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f" | GPU: {mem:.1f}GB")
        else:
            print()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Вызывается в конце каждой эпохи"""
        epoch_time = time.time() - self.epoch_start_time
        loss = last_logged_loss(state.log_history)
        print(f"Эпоха {state.epoch} завершена | Время: {epoch_time/60:.1f}мин | Loss: {format_metric(loss, '.4f')}\n")
        
    def on_train_end(self, args, state, control, **kwargs):
        """Вызывается в конце обучения"""
        total_time = time.time() - self.start_time
        loss = last_logged_loss(state.log_history)
        print(f"\n{'='*60}")
        print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"Время: {total_time/60:.1f}мин | Шагов: {state.global_step} | Loss: {format_metric(loss, '.4f')}")
        print(f"{'='*60}\n")

def print_system_info():
    """Выводит информацию о системе"""
    print("\n" + "="*60)
    print("СИСТЕМА")
    print("="*60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"CUDA: {cuda_version}")
        print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        
        # Проверка совместимости для RTX 5060 (sm_120)
        if compute_capability[0] >= 12:
            print("⚠ RTX 5060 (sm_120) требует PyTorch 2.5+ или nightly build")
            print("Если возникают ошибки, установите:")
            print("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
    else:
        print("[WARN] ВНИМАНИЕ: CUDA недоступна!")
        print("PyTorch установлен без поддержки GPU")
        print("Обучение будет выполняться на CPU")
    print("="*60)

def load_model_and_tokenizer(model_name, use_4bit=True, cache_dir=None, device="auto"):
    """
    Загружает модель и токенизатор
    
    Args:
        model_name: имя модели с HuggingFace
        use_4bit: использовать ли 4-bit quantization для экономии памяти
        cache_dir: директория для сохранения модели (если None, используется локальная директория)
    """
    print(f"\n{'='*60}")
    print(f"ЗАГРУЗКА МОДЕЛИ: {model_name}")
    print(f"{'='*60}")
    use_gpu = device != "cpu" and torch.cuda.is_available()

    if use_4bit and not use_gpu:
        print("[WARN] 4-bit quantization доступна только на GPU. Отключаем --use_4bit для CPU.")
        use_4bit = False

    if use_4bit:
        print("Quantization: 4-bit")
    
    # Определяем директорию для сохранения модели
    if cache_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "models", model_name.replace("/", "_"))
        os.makedirs(cache_dir, exist_ok=True)
    else:
        os.makedirs(cache_dir, exist_ok=True)
    
    load_start = time.time()
    
    # Настройка quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Загрузка токенизатора
    if os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        print("Токенизатор: локальный")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.save_pretrained(cache_dir)
        print("Токенизатор: скачан")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Загрузка модели
    model_start = time.time()
    if use_gpu:
        device_map = "auto"
        torch_dtype = torch.float16 if not use_4bit else None
        torch.cuda.set_device(0)

        compute_cap = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Устройство: GPU ({gpu_name})")

        if compute_cap[0] >= 12:
            print(f"[WARN] Обнаружена архитектура sm_{compute_cap[0]}{compute_cap[1]} (Blackwell)")
            print("  Для RTX 5060 требуется PyTorch 2.5+ или newer build")
            if use_4bit:
                print("  [WARN] Quantization может не работать с sm_120")
                print("  Если возникнут ошибки, запустите БЕЗ --use_4bit")
    else:
        device_map = None
        torch_dtype = torch.float32
        print("Устройство: CPU")
    
    # Проверяем, есть ли уже сохраненная полная модель локально
    has_local_model = has_complete_local_model(cache_dir)

    # Попытка загрузки с обработкой ошибок quantization
    try:
        if has_local_model:
            if use_4bit:
                quantization_config_path = os.path.join(cache_dir, "quantization_config.json")
                if not os.path.exists(quantization_config_path):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=bnb_config,
                        device_map=device_map
                    )
                    if hasattr(model, 'config'):
                        model.config.save_pretrained(cache_dir)
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        cache_dir,
                        quantization_config=bnb_config,
                        device_map=device_map
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    cache_dir,
                    device_map=device_map,
                    torch_dtype=torch_dtype
                )
        else:
            # Попытка загрузки с обработкой ошибок quantization для sm_120
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config if use_4bit else None,
                    device_map=device_map,
                    torch_dtype=torch_dtype
                )
                if not use_4bit:
                    model.save_pretrained(cache_dir)
                else:
                    if hasattr(model, 'config'):
                        model.config.save_pretrained(cache_dir)
            except RuntimeError as e:
                error_str = str(e)
                if "no kernel image is available" in error_str or "CUDA capability" in error_str or "sm_120" in error_str:
                    print("\n" + "="*60)
                    print("ОШИБКА: Несовместимость с RTX 5060 (sm_120)")
                    print("="*60)
                    print("Проблема: bitsandbytes не поддерживает sm_120")
                    print("\nРешение 1: Установите PyTorch Nightly (рекомендуется)")
                    print("  pip uninstall torch torchvision torchaudio bitsandbytes")
                    print("  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                    print("  pip install bitsandbytes")
                    print("\nРешение 2: Запустите БЕЗ quantization")
                    print("  python train.py --model_name ... --dataset_path ...")
                    print("  (уберите флаг --use_4bit)")
                    print("="*60)
                    sys.exit(1)
                else:
                    raise
    except RuntimeError as e:
        if use_gpu and ("no kernel image is available" in str(e) or "CUDA capability" in str(e)):
            print("\n" + "="*60)
            print("ОШИБКА: Несовместимость CUDA capability")
            print("="*60)
            print("RTX 5060 (sm_120) требует PyTorch 2.5+ или nightly build")
            print("\nРешение 1: Установите PyTorch Nightly")
            print("pip uninstall torch torchvision torchaudio bitsandbytes")
            print("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
            print("pip install bitsandbytes")
            print("\nРешение 2: Запустите БЕЗ quantization (медленнее, но работает)")
            print("python train.py --model_name ... --dataset_path ...")
            print("(уберите флаг --use_4bit)")
            print("="*60)
            sys.exit(1)
        else:
            raise
    
    model_time = time.time() - model_start
    
    # Информация о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметры: {total_params/1e6:.1f}M всего, {trainable_params/1e6:.1f}M обучаемых")
    
    # Память GPU
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated(0) / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU память: {mem:.1f}/{total_mem:.1f}GB")
    
    # Подготовка модели для обучения с quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    total_load_time = time.time() - load_start
    print(f"Загружено за {total_load_time:.1f}с\n")
    
    return model, tokenizer

def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Настраивает LoRA для модели
    
    Args:
        model: модель для настройки
        r: rank LoRA
        lora_alpha: alpha параметр LoRA
        lora_dropout: dropout для LoRA
    """
    print(f"{'='*60}")
    print(f"НАСТРОЙКА LoRA | r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"{'='*60}")
    
    # Определяем target_modules в зависимости от архитектуры модели
    model_type = model.config.model_type.lower() if hasattr(model.config, 'model_type') else 'unknown'
    
    # Определяем target_modules в зависимости от архитектуры
    if model_type in ['gpt2', 'gpt_neo', 'gpt_neo_x']:
        # GPT-2, DialoGPT, GPT-Neo используют c_attn и c_proj
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif model_type in ['llama', 'mistral', 'mixtral']:
        # LLaMA модели используют q_proj, v_proj и т.д.
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif model_type in ['bloom', 'bloomz']:
        # BLOOM модели
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif model_type in ['opt']:
        # OPT модели
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    else:
        # Пытаемся автоматически найти подходящие модули
        print("  Автоматическое определение модулей...")
        all_module_names = set()
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Листовые модули
                module_name = name.split('.')[-1]
                all_module_names.add(module_name)
        
        # Ищем распространенные паттерны
        target_modules = []
        common_patterns = ['attn', 'proj', 'fc', 'dense', 'query', 'key', 'value']
        for pattern in common_patterns:
            matching = [name for name in all_module_names if pattern.lower() in name.lower()]
            target_modules.extend(matching)
        
        if not target_modules:
            # Если ничего не найдено, используем все линейные слои
            target_modules = [name for name in all_module_names if 'linear' in name.lower() or 'weight' in name.lower()]
        
        if not target_modules:
            # Последняя попытка - используем стандартные для GPT-2
            print("  [WARN] Не удалось определить модули, используем стандартные для GPT-2")
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            target_modules = list(set(target_modules))  # Убираем дубликаты
    
    print(f"Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Информация о параметрах
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = (trainable_params / total_params) * 100
    print(f"Обучаемых параметров: {trainable_params/1e6:.2f}M ({trainable_percent:.2f}%)\n")
    
    return model

def find_dataset_schema(record):
    """Возвращает первую схему из DATASET_SCHEMAS, все поля которой есть в записи, иначе None."""
    for schema in DATASET_SCHEMAS:
        if all(key in record for key in schema):
            return schema
    return None

def validate_record(record, where):
    """
    Проверяет одну запись датасета: объект, поддерживаемая схема, обязательные поля - непустые строки.
    where - расположение записи для сообщения об ошибке (файл, строка/номер записи).
    """
    if not isinstance(record, dict):
        raise ValueError(f"{where}: запись должна быть JSON-объектом, получено: {type(record).__name__}")

    schema = find_dataset_schema(record)
    if schema is None:
        supported = "; ".join(" + ".join(s) for s in DATASET_SCHEMAS)
        raise ValueError(
            f"{where}: неподдерживаемая запись (ключи: {sorted(record)}). Поддерживаются схемы: {supported}"
        )

    for key in schema:
        value = record[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{where}: поле '{key}' должно быть непустой строкой, получено: {value!r}")

def format_example(example):
    """Собирает текст обучающего примера по поддерживаемой схеме (запись должна быть валидной)."""
    schema = find_dataset_schema(example)
    if schema is None:
        raise ValueError(f"Неподдерживаемая запись датасета (ключи: {sorted(example)})")
    return DATASET_SCHEMAS[schema](example)

def load_dataset_from_file(dataset_path):
    """
    Загружает и строго проверяет датасет из файла

    Поддерживаемые форматы:
    - JSON: список записей ИЛИ объект с ключом 'data', содержащим список записей
    - JSONL: каждая непустая строка - JSON-объект (пустые строки пропускаются)

    Запись - объект с одной из схем из DATASET_SCHEMAS ('text'; 'instruction'+'output';
    'prompt'+'completion'; 'input'+'output'), обязательные поля - непустые строки.
    Некорректный JSON, неподдерживаемая запись или пустой датасет - ValueError с указанием места.
    """
    print(f"{'='*60}")
    print(f"ЗАГРУЗКА ДАТАСЕТА: {os.path.basename(dataset_path)}")
    print(f"{'='*60}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл датасета не найден: {dataset_path}")

    load_start = time.time()
    file_name = os.path.basename(dataset_path)

    # utf-8-sig: файлы из Windows-редакторов могут начинаться с BOM
    if dataset_path.endswith('.jsonl'):
        data = []
        with open(dataset_path, 'r', encoding='utf-8-sig') as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                where = f"{file_name}, строка {line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{where}: некорректный JSON ({exc.msg}, столбец {exc.colno})") from exc
                validate_record(record, where)
                data.append(record)
    elif dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8-sig') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{file_name}: некорректный JSON ({exc.msg}, строка {exc.lineno}, столбец {exc.colno})"
                ) from exc
        if isinstance(data, dict):
            if 'data' not in data:
                raise ValueError(
                    f"{file_name}: ожидается список записей или объект с ключом 'data', "
                    f"получен объект с ключами: {sorted(data)}"
                )
            data = data['data']
        if not isinstance(data, list):
            raise ValueError(f"{file_name}: коллекция записей должна быть списком, получено: {type(data).__name__}")
        for record_number, record in enumerate(data, start=1):
            validate_record(record, f"{file_name}, запись {record_number}")
    else:
        raise ValueError("Поддерживаются только .json и .jsonl файлы")

    if not data:
        raise ValueError(f"{file_name}: датасет пуст, нет ни одной записи")

    load_time = time.time() - load_start
    file_size = os.path.getsize(dataset_path) / 1024**2
    print(f"Примеров: {len(data)} | Размер: {file_size:.1f}MB | Время: {load_time:.1f}с\n")
    
    return data

def preprocess_dataset(data, tokenizer, max_length=512):
    """
    Предобрабатывает датасет для обучения
    
    Args:
        data: список проверенных записей (см. load_dataset_from_file)
        tokenizer: токенизатор
        max_length: максимальная длина последовательности
    """
    print(f"{'='*60}")
    print(f"ПРЕДОБРАБОТКА | max_length={max_length}")
    print(f"{'='*60}")

    preprocess_start = time.time()

    # Сразу приводим все записи к тексту: так записи с разными схемами и лишними полями
    # не портят таблицу datasets
    texts = [format_example(example) for example in data]

    # Анализ длины текстов (только статистика)
    text_lengths = []
    for text in texts[:100]:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        text_lengths.append(len(tokens))
    
    if text_lengths:
        avg_tokens = sum(text_lengths) / len(text_lengths)
        truncated = sum(1 for t in text_lengths if t + 1 > max_length)
        print(f"Средняя длина: {avg_tokens:.0f} токенов | Обрезано: {truncated}/{len(text_lengths)}")

    if tokenizer.eos_token_id is None:
        raise ValueError("Токенизатор должен иметь eos_token_id для завершения обучающих примеров")
    
    def tokenize_function(examples):
        # Резервируем последний слот под EOS; padding добавит collator до длины текущего батча.
        if max_length == 1:
            input_ids = [[] for _ in examples["text"]]
        else:
            input_ids = tokenizer(
                examples["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length - 1,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]

        input_ids = [ids + [tokenizer.eos_token_id] for ids in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }
    
    # Преобразуем в формат для datasets
    dataset = Dataset.from_list([{"text": text} for text in texts])
    
    # Токенизация
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Токенизация"
    )
    
    preprocess_time = time.time() - preprocess_start
    total_tokens = sum(len(ids) for ids in tokenized_dataset['input_ids'])
    print(f"Токенизировано: {len(tokenized_dataset)} примеров, {total_tokens/1e3:.0f}K токенов | Время: {preprocess_time:.1f}с\n")
    
    return tokenized_dataset

def train(
    model_name,
    dataset_path,
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=512,
    use_4bit=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    save_steps=500,
    logging_steps=10,
    warmup_steps=100,
    device="auto",
    seed=42
):
    """
    Основная функция обучения
    
    Args:
        model_name: имя модели с HuggingFace
        dataset_path: путь к датасету
        output_dir: директория для сохранения модели
        num_train_epochs: количество эпох
        per_device_train_batch_size: размер батча на устройство
        gradient_accumulation_steps: шаги накопления градиента
        learning_rate: скорость обучения
        max_length: максимальная длина последовательности
        use_4bit: использовать ли 4-bit quantization
        lora_r: rank LoRA
        lora_alpha: alpha параметр LoRA
        lora_dropout: dropout для LoRA
        save_steps: шаги сохранения
        logging_steps: шаги логирования
        warmup_steps: шаги warmup
        seed: seed для воспроизводимости (инициализация LoRA, порядок данных, dropout)
    """
    # Числовые параметры проверяем первыми - до загрузки датасета и модели
    validate_train_args(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        save_steps=save_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        seed=seed,
    )

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device должен быть одним из: auto, cpu, cuda")

    use_gpu = device != "cpu" and torch.cuda.is_available()
    resolved_device = "cuda:0" if use_gpu else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Запрошен device=cuda, но CUDA недоступна")

    if use_4bit and not use_gpu:
        print("[WARN] 4-bit quantization на CPU не поддерживается. Отключаем --use_4bit.")
        use_4bit = False

    # Вывод информации о системе
    print_system_info()
    print(f"Выбранное устройство: {resolved_device}")

    # Датасет читаем и проверяем до загрузки модели: ошибка в данных не должна стоить загрузки весов
    data = load_dataset_from_file(dataset_path)

    # Seed до загрузки модели: он влияет на случайную инициализацию LoRA-матриц
    set_seed(seed)

    # Определяем директорию для сохранения обученной модели (в проекте)
    if not os.path.isabs(output_dir):
        # Если путь относительный, делаем его относительно корня проекта
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, output_dir)
    
    # Создаем директорию для сохранения модели
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nДиректория для сохранения обученной модели: {os.path.abspath(output_dir)}")
    
    # Определяем директорию для базовой модели (в проекте)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_cache_dir = os.path.join(project_root, "models", model_name.replace("/", "_"))
    
    # Загрузка модели и токенизатора
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        use_4bit=use_4bit,
        cache_dir=base_model_cache_dir,
        device="cuda" if use_gpu else "cpu"
    )
    
    # Настройка LoRA
    model = setup_lora(model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    try:
        model_device = next(model.parameters()).device
        if use_gpu and model_device.type == 'cpu':
            print("[WARN] Модель на CPU, перемещаем на GPU...")
            model = model.to(resolved_device)
        elif not use_gpu and model_device.type != 'cpu':
            print("[WARN] Модель не на CPU, перемещаем на CPU...")
            model = model.to("cpu")
        else:
            print(f"[OK] Модель на устройстве: {model_device}")
    except Exception:
        model = model.to(resolved_device)
        print(f"[OK] Модель перемещена на {resolved_device}")
    
    # Токенизация датасета (сам датасет уже загружен и проверен выше)
    train_dataset = preprocess_dataset(data, tokenizer, max_length=max_length)

    data_collator = CausalLMCollator(tokenizer)

    # Вычисляем количество шагов так же, как Trainer (с учётом неполных батчей)
    steps_per_epoch, total_steps = compute_training_steps(
        len(train_dataset), per_device_train_batch_size, gradient_accumulation_steps, num_train_epochs
    )

    print("\n" + "="*80)
    print("НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ")
    print("="*80)
    print(f"Размер датасета: {len(train_dataset)} примеров")
    print(f"Размер батча на устройство: {per_device_train_batch_size}")
    print(f"Шаги накопления градиента: {gradient_accumulation_steps}")
    print(f"Эффективный размер батча: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"Шагов в эпохе: {steps_per_epoch}")
    print(f"Всего шагов: {total_steps}")
    print(f"Количество эпох: {num_train_epochs}")
    print(f"Скорость обучения: {learning_rate}")
    print(f"Warmup шагов: {warmup_steps}")
    print(f"Шаги логирования: {logging_steps}")
    print(f"Шаги сохранения: {save_steps}")
    print(f"Seed: {seed}")
    use_gradient_checkpointing = False  # Инициализация
    
    # Проверка конкретной GPU модели
    gpu_name = None
    gpu_memory_gb = None
    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # RTX 5060 имеет ~8GB VRAM, оптимизируем настройки
        if "RTX 5060" in gpu_name or gpu_memory_gb < 10:
            use_gradient_checkpointing = True
            if per_device_train_batch_size > 4:
                print(f"[WARN] Батч {per_device_train_batch_size} может быть слишком большим для 8GB GPU")
        else:
            use_gradient_checkpointing = False
    
    if use_4bit and use_gpu:
        optim_name = "paged_adamw_8bit"
    elif use_gpu:
        optim_name = "adamw_torch"
    else:
        optim_name = "adamw_torch"
    
    # Настройки для GPU
    if use_gpu:
        fp16 = True
        bf16 = False
        dataloader_pin_memory = True
        dataloader_num_workers = 4
    else:
        fp16 = False
        bf16 = False
        dataloader_pin_memory = False
        dataloader_num_workers = 0
        use_gradient_checkpointing = False
        print("[WARN] Обучение на CPU будет очень медленным!\n")
    
    # Включаем gradient checkpointing если нужно
    if use_gpu and use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=fp16,  # FP16 для RTX 5060
        bf16=bf16,  # BF16 не поддерживается RTX 5060
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_num_workers=dataloader_num_workers,
        logging_steps=logging_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        seed=seed,  # порядок данных и dropout; data_seed по умолчанию берётся из seed
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        use_cpu=not use_gpu,
        optim=optim_name,
        logging_first_step=True,
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,  # Сохраняем все колонки
        ddp_find_unused_parameters=False if use_gpu else None,  # Оптимизация для multi-GPU
        gradient_checkpointing=use_gradient_checkpointing if use_gpu else False,  # Экономия памяти
    )
    
    # Создаем директорию для логов
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Trainer с callback для детального логирования
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[DetailedLoggingCallback(steps_per_epoch)],
    )
    
    # Убеждаемся что модель на нужном устройстве перед обучением
    try:
        model_device = next(model.parameters()).device
        target_device_type = "cuda" if use_gpu else "cpu"
        if model_device.type != target_device_type:
            print(f"Перемещение модели на {resolved_device} перед обучением...")
            model = model.to(resolved_device)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                callbacks=[DetailedLoggingCallback(steps_per_epoch)],
            )
    except Exception:
        model = model.to(resolved_device)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[DetailedLoggingCallback(steps_per_epoch)],
        )
    
    # Информация о памяти перед обучением
    if use_gpu:
        mem = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - mem
        print(f"GPU память: {mem:.1f}/{total:.1f}GB (свободно: {free:.1f}GB)")
        if free < 1.0:
            print("[WARN] Мало памяти! Уменьшите batch_size")
        torch.cuda.empty_cache()
    else:
        print("Обучение запускается на CPU. Это может быть заметно медленнее.")
    print()
    
    # Обучение
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    # Финальная статистика
    loss = last_logged_loss(trainer.state.log_history)
    speed = len(train_dataset) * num_train_epochs / train_time if train_time > 0 else 0
    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Время: {train_time/60:.1f}мин | Скорость: {speed:.1f} примеров/сек | Loss: {format_metric(loss, '.4f')}")
    print(f"{'='*60}\n")
    
    # Сохранение модели
    print(f"Сохранение модели в {output_dir}...")
    save_start = time.time()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_time = time.time() - save_start
    print(f"[OK] Сохранено за {save_time:.1f}с | Путь: {os.path.abspath(output_dir)}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tuning модели с LoRA")
    parser.add_argument("--model_name", type=str, required=True, help="Имя модели с HuggingFace")
    parser.add_argument("--dataset_path", type=str, required=True, help="Путь к датасету (.json или .jsonl)")
    parser.add_argument("--output_dir", type=str, default="./lora_model", help="Директория для сохранения")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Шаги накопления градиента")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Скорость обучения")
    parser.add_argument("--max_length", type=int, default=512, help="Максимальная длина последовательности")
    parser.add_argument("--use_4bit", action="store_true", help="Использовать 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16, help="Rank LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout LoRA")
    parser.add_argument("--save_steps", type=int, default=500, help="Шаги сохранения")
    parser.add_argument("--logging_steps", type=int, default=10, help="Шаги логирования")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Шаги warmup")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Устройство для обучения")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости (по умолчанию 42)")

    args = parser.parse_args(argv)

    try:
        train(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            use_4bit=args.use_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            warmup_steps=args.warmup_steps,
            device=args.device,
            seed=args.seed,
        )
    except InvalidArguments as exc:
        # train() проверяет аргументы до любой тяжёлой работы, поэтому ошибка приходит сразу
        parser.error(str(exc))

if __name__ == "__main__":
    main()

