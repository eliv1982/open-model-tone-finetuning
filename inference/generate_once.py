"""
Однократная генерация в формате обучения: ### Instruction / ### Response.
Для демонстрации LoRA без интерактивного чата User:/Assistant:.
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from preflight import has_complete_local_model, validate_generation_args, validate_lora_adapter


def build_prompt(instruction):
    """Промпт в том же формате, что при обучении (train.py)."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def load_model_and_tokenizer(base_model_name, lora_model_path=None):
    """
    Загружает базовую модель (локально или с HF) и опционально объединяет LoRA.
    Если lora_model_path задан, адаптер обязан загрузиться: иначе исключение, а не baseline.
    """
    if lora_model_path is not None:
        validate_lora_adapter(lora_model_path)

    if torch.cuda.is_available():
        torch_dtype = torch.float16
        device_map = "auto"
        device = torch.device("cuda")
    else:
        torch_dtype = torch.float32
        device_map = None
        device = torch.device("cpu")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_cache_dir = os.path.join(
        project_root, "models", base_model_name.replace("/", "_")
    )
    os.makedirs(base_model_cache_dir, exist_ok=True)

    has_local_model = has_complete_local_model(base_model_cache_dir)
    model_path = base_model_cache_dir if has_local_model else base_model_name

    tokenizer_config_path = os.path.join(base_model_cache_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        tokenizer = AutoTokenizer.from_pretrained(base_model_cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(base_model_cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if not has_local_model:
        model.save_pretrained(base_model_cache_dir)

    if device_map is None:
        model = model.to(device)

    if lora_model_path is not None:
        model = PeftModel.from_pretrained(model, lora_model_path)
        model = model.merge_and_unload()
        if device_map is None:
            model = model.to(device)

    model.eval()
    return model, tokenizer, device


def generate_once(
    model,
    tokenizer,
    device,
    formatted_prompt,
    max_new_tokens,
    temperature,
    top_p,
    seed=42,
):
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if hasattr(model, "device"):
        target_device = model.device
    else:
        target_device = device
    input_ids = inputs["input_ids"].to(target_device)
    attention_mask = inputs["attention_mask"].to(target_device)
    input_length = input_ids.shape[1]

    set_seed(seed)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Один запрос к модели в формате ### Instruction / ### Response"
    )
    parser.add_argument("--base_model", type=str, required=True, help="Базовая модель Hugging Face")
    parser.add_argument(
        "--lora_model",
        type=str,
        default=None,
        help="Путь к LoRA-адаптеру. Если задан, но адаптер не найден/некорректен - ошибка, а не baseline",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Текст instruction (вопрос)")
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Макс. новых токенов")
    parser.add_argument("--temperature", type=float, default=0.2, help="Температура сэмплирования")
    parser.add_argument("--top_p", type=float, default=0.85, help="Nucleus sampling top_p")
    parser.add_argument("--seed", type=int, default=42, help="Seed сэмплирования (по умолчанию 42)")
    args = parser.parse_args(argv)

    # Все проверки - до загрузки модели
    try:
        validate_generation_args("max_new_tokens", args.max_new_tokens, args.temperature, args.top_p)
        if args.lora_model is not None:
            validate_lora_adapter(args.lora_model)
    except ValueError as exc:
        parser.error(str(exc))

    formatted_prompt = build_prompt(args.prompt)

    model, tokenizer, device = load_model_and_tokenizer(args.base_model, args.lora_model)

    answer = generate_once(
        model,
        tokenizer,
        device,
        formatted_prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.seed,
    )

    print(f"Базовая модель: {args.base_model}")
    # Сюда доходим, только если адаптер реально загружен (иначе выше было бы исключение)
    if args.lora_model is not None:
        print(f"LoRA: {args.lora_model} (адаптер загружен и объединён с базовой моделью)")
    else:
        print("LoRA: не используется (baseline)")
    print(f"Вопрос: {args.prompt}")
    print(f"Ответ: {answer}")


if __name__ == "__main__":
    main()
