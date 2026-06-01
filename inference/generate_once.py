"""
Однократная генерация в формате обучения: ### Instruction / ### Response.
Для демонстрации LoRA без интерактивного чата User:/Assistant:.
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(base_model_name, lora_model_path=None):
    """Загружает базовую модель (локально или с HF) и опционально объединяет LoRA."""
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

    has_local_model = os.path.exists(os.path.join(base_model_cache_dir, "config.json"))
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
        trust_remote_code=True,
    )

    if not has_local_model:
        model.save_pretrained(base_model_cache_dir)

    if device_map is None:
        model = model.to(device)

    if lora_model_path and os.path.exists(lora_model_path):
        model = PeftModel.from_pretrained(model, lora_model_path)
        model = model.merge_and_unload()
        if device_map is None:
            model = model.to(device)
    elif lora_model_path:
        print(f"Предупреждение: путь LoRA не найден: {lora_model_path}. Используется базовая модель.")

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


def main():
    parser = argparse.ArgumentParser(
        description="Один запрос к модели в формате ### Instruction / ### Response"
    )
    parser.add_argument("--base_model", type=str, required=True, help="Базовая модель Hugging Face")
    parser.add_argument("--lora_model", type=str, default=None, help="Путь к LoRA-адаптеру")
    parser.add_argument("--prompt", type=str, required=True, help="Текст instruction (вопрос)")
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Макс. новых токенов")
    parser.add_argument("--temperature", type=float, default=0.2, help="Температура сэмплирования")
    parser.add_argument("--top_p", type=float, default=0.85, help="Nucleus sampling top_p")
    args = parser.parse_args()

    formatted_prompt = f"### Instruction:\n{args.prompt}\n\n### Response:\n"

    model, tokenizer, device = load_model_and_tokenizer(args.base_model, args.lora_model)

    answer = generate_once(
        model,
        tokenizer,
        device,
        formatted_prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    print(f"Базовая модель: {args.base_model}")
    if args.lora_model:
        print(f"LoRA: {args.lora_model}")
    else:
        print("LoRA: не используется")
    print(f"Вопрос: {args.prompt}")
    print(f"Ответ: {answer}")


if __name__ == "__main__":
    main()
