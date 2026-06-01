"""
Скрипт для запуска модели в терминале для общения
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse

def load_model_and_tokenizer(base_model_name, lora_model_path=None):
    """
    Загружает модель и токенизатор
    
    Args:
        base_model_name: имя базовой модели с HuggingFace
        lora_model_path: путь к дообученной LoRA модели (опционально)
    """
    print(f"Загрузка модели {base_model_name}...")
    
    # Проверка наличия GPU
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"\n{'='*60}")
        print(f"ИСПОЛЬЗОВАНИЕ GPU")
        print(f"{'='*60}")
        print(f"GPU устройство: {device_name}")
        print(f"Количество GPU: {device_count}")
        print(f"Текущий GPU: {current_device}")
        
        # Показываем информацию о памяти GPU
        for i in range(device_count):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i} память: {total_memory:.2f} GB")
        print(f"{'='*60}\n")
        
        torch_dtype = torch.float16
        device_map = "auto"
    else:
        device = "cpu"
        torch_dtype = torch.float32
        device_map = None
        print(f"\n⚠ ВНИМАНИЕ: GPU не обнаружен, используется CPU")
        print(f"Генерация на CPU будет медленной!\n")
    
    # Определяем директорию для базовой модели (в проекте)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_cache_dir = os.path.join(project_root, "models", base_model_name.replace("/", "_"))
    os.makedirs(base_model_cache_dir, exist_ok=True)
    
    config_path = os.path.join(base_model_cache_dir, "config.json")
    has_local_model = os.path.exists(config_path)
    
    # Проверяем, есть ли локальная копия модели
    if has_local_model:
        print(f"Использование локальной копии модели из {base_model_cache_dir}")
        model_path = base_model_cache_dir
    else:
        print(f"Модель будет скачана и сохранена в {base_model_cache_dir}")
        model_path = base_model_name
    
    # Загрузка токенизатора
    tokenizer_config_path = os.path.join(base_model_cache_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        print(f"Загрузка токенизатора из локальной директории...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_cache_dir)
    else:
        print(f"Загрузка токенизатора из HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(base_model_cache_dir)
        print(f"Токенизатор сохранен в {base_model_cache_dir}")
    
    # Установка pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Загрузка базовой модели
    print(f"Загрузка модели из {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # Сохраняем модель локально, если она была скачана
    if not has_local_model:
        print(f"Сохранение модели в {base_model_cache_dir}...")
        model.save_pretrained(base_model_cache_dir)
        print(f"Модель сохранена!")
    
    # Показываем информацию о памяти GPU после загрузки
    if torch.cuda.is_available():
        print(f"\nСостояние памяти GPU после загрузки:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = total - reserved
            print(f"  GPU {i}: {allocated:.2f}GB / {reserved:.2f}GB (свободно: {free:.2f}GB)")
    
    # Загрузка LoRA весов если указан путь
    if lora_model_path and os.path.exists(lora_model_path):
        print(f"Загрузка LoRA весов из {lora_model_path}...")
        model = PeftModel.from_pretrained(model, lora_model_path)
        model = model.merge_and_unload()  # Объединяем LoRA веса с базовой моделью
        print("LoRA веса успешно загружены и объединены!")
    elif lora_model_path:
        print(f"Предупреждение: Путь {lora_model_path} не найден. Используется базовая модель.")
    
    # Переводим модель в режим оценки
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9, top_k=50):
    """
    Генерирует ответ модели на промпт
    
    Args:
        model: модель
        tokenizer: токенизатор
        prompt: входной промпт
        max_length: максимальная длина генерируемого текста
        temperature: температура для генерации (чем выше, тем более случайно)
        top_p: nucleus sampling параметр
        top_k: top-k sampling параметр
    """
    # Токенизация промпта
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Перемещение на устройство модели (автоматически определяется device_map)
    # Если device_map="auto", модель сама определяет устройство
    if hasattr(model, 'device'):
        inputs = inputs.to(model.device)
    elif torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Декодирование ответа
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Удаляем промпт из ответа, оставляем только сгенерированный текст
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

def chat_loop(model, tokenizer, system_prompt="", max_length=512, temperature=0.7):
    """
    Основной цикл чата
    
    Args:
        model: модель
        tokenizer: токенизатор
        system_prompt: системный промпт (опционально)
        max_length: максимальная длина ответа
        temperature: температура генерации
    """
    print("\n" + "="*50)
    print("Чат с моделью запущен!")
    print("Введите 'quit', 'exit' или 'q' для выхода")
    print("Введите 'clear' для очистки истории")
    print("="*50 + "\n")
    
    conversation_history = []
    
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})
    
    while True:
        try:
            # Получение ввода пользователя
            user_input = input("Вы: ").strip()
            
            if not user_input:
                continue
            
            # Команды выхода
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("До свидания!")
                break
            
            # Очистка истории
            if user_input.lower() == 'clear':
                conversation_history = []
                if system_prompt:
                    conversation_history.append({"role": "system", "content": system_prompt})
                print("История очищена.\n")
                continue
            
            # Формирование промпта из истории разговора
            if conversation_history:
                # Форматируем историю для модели
                prompt_parts = []
                for msg in conversation_history:
                    if msg["role"] == "system":
                        prompt_parts.append(f"System: {msg['content']}")
                    elif msg["role"] == "user":
                        prompt_parts.append(f"User: {msg['content']}")
                    elif msg["role"] == "assistant":
                        prompt_parts.append(f"Assistant: {msg['content']}")
                
                prompt_parts.append(f"User: {user_input}")
                prompt_parts.append("Assistant:")
                prompt = "\n".join(prompt_parts)
            else:
                prompt = f"User: {user_input}\nAssistant:"
            
            # Генерация ответа
            print("Модель генерирует ответ...")
            response = generate_response(
                model, 
                tokenizer, 
                prompt, 
                max_length=max_length,
                temperature=temperature
            )
            
            # Вывод ответа
            print(f"\nМодель: {response}\n")
            
            # Сохранение в историю
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            
            # Ограничение истории (оставляем последние 10 сообщений)
            if len(conversation_history) > 20:
                if system_prompt:
                    conversation_history = [conversation_history[0]] + conversation_history[-19:]
                else:
                    conversation_history = conversation_history[-20:]
        
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")
            continue

def main():
    parser = argparse.ArgumentParser(description="Чат с моделью в терминале")
    parser.add_argument("--base_model", type=str, required=True, help="Имя базовой модели с HuggingFace")
    parser.add_argument("--lora_model", type=str, default=None, help="Путь к дообученной LoRA модели")
    parser.add_argument("--system_prompt", type=str, default="", help="Системный промпт")
    parser.add_argument("--max_length", type=int, default=512, help="Максимальная длина ответа")
    parser.add_argument("--temperature", type=float, default=0.7, help="Температура генерации")
    
    args = parser.parse_args()
    
    # Загрузка модели
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.lora_model)
    
    # Запуск чата
    chat_loop(
        model, 
        tokenizer, 
        system_prompt=args.system_prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()

