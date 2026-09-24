import json

import pytest
import torch

import chat
import generate_once
import preflight
import train

BASE_MODEL = "test-org/tiny-model"


# --- помощники -------------------------------------------------------------

def make_adapter(directory, weights="adapter_model.safetensors", peft_type="LORA"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({"peft_type": peft_type, "r": 8}), encoding="utf-8")
    if weights:
        (directory / weights).write_bytes(b"\0")
    return str(directory)


class FakeModel:
    device = "cpu"

    def __init__(self, merged=False):
        self.merged = merged

    def to(self, _device):
        return self

    def eval(self):
        return self

    def save_pretrained(self, _path):
        pass

    def merge_and_unload(self):
        return FakeModel(merged=True)


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def save_pretrained(self, _path):
        pass


class Recorder:
    """Подмена AutoModelForCausalLM / AutoTokenizer / PeftModel: запоминает вызовы, ничего не скачивает."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def from_pretrained(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture(params=[generate_once, chat], ids=["generate_once", "chat"])
def script(request):
    return request.param


@pytest.fixture
def mocked_loading(script, tmp_path, monkeypatch):
    """Подменяет загрузку моделей; кэш моделей скрипта переносится в tmp_path/models."""
    monkeypatch.setattr(script, "__file__", str(tmp_path / "inference" / "script.py"))
    model = Recorder(FakeModel())
    tokenizer = Recorder(FakeTokenizer())
    peft = Recorder(FakeModel())
    monkeypatch.setattr(script, "AutoModelForCausalLM", model)
    monkeypatch.setattr(script, "AutoTokenizer", tokenizer)
    monkeypatch.setattr(script.PeftModel, "from_pretrained", peft.from_pretrained)
    cache_dir = tmp_path / "models" / BASE_MODEL.replace("/", "_")
    return {"model": model, "tokenizer": tokenizer, "peft": peft, "cache_dir": cache_dir}


# --- кэш модели ------------------------------------------------------------

def test_cache_missing_dir_is_incomplete(tmp_path):
    assert preflight.has_complete_local_model(str(tmp_path / "nope")) is False


def test_cache_empty_dir_is_incomplete(tmp_path):
    assert preflight.has_complete_local_model(str(tmp_path)) is False


def test_cache_with_config_only_is_incomplete(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    assert preflight.has_complete_local_model(str(tmp_path)) is False


def test_cache_with_weights_but_no_config_is_incomplete(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\0")
    assert preflight.has_complete_local_model(str(tmp_path)) is False


@pytest.mark.parametrize("weight_file", preflight.MODEL_WEIGHT_FILES)
def test_cache_with_config_and_weights_is_complete(tmp_path, weight_file):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / weight_file).write_bytes(b"\0")
    assert preflight.has_complete_local_model(str(tmp_path)) is True


def test_train_cache_check_matches_inference(tmp_path):
    assert train.MODEL_WEIGHT_FILES == preflight.MODEL_WEIGHT_FILES
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin.index.json").write_text("{}")
    assert train.has_complete_local_model(str(tmp_path)) is preflight.has_complete_local_model(str(tmp_path)) is True


def test_incomplete_cache_falls_back_to_hub_name(script, mocked_loading):
    cache_dir = mocked_loading["cache_dir"]
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_text("{}")

    script.load_model_and_tokenizer(BASE_MODEL)

    (args, _kwargs), = mocked_loading["model"].calls
    assert args[0] == BASE_MODEL


def test_complete_cache_is_used_as_model_path(script, mocked_loading):
    cache_dir = mocked_loading["cache_dir"]
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_text("{}")
    (cache_dir / "model.safetensors").write_bytes(b"\0")

    script.load_model_and_tokenizer(BASE_MODEL)

    (args, _kwargs), = mocked_loading["model"].calls
    assert args[0] == str(cache_dir)


def test_trust_remote_code_is_never_requested(script, mocked_loading):
    script.load_model_and_tokenizer(BASE_MODEL)

    for recorder in (mocked_loading["model"], mocked_loading["tokenizer"]):
        for _args, kwargs in recorder.calls:
            assert "trust_remote_code" not in kwargs


# --- строгая обработка LoRA ------------------------------------------------

def test_valid_adapter_passes_validation(tmp_path):
    preflight.validate_lora_adapter(make_adapter(tmp_path / "a"))
    preflight.validate_lora_adapter(make_adapter(tmp_path / "b", weights="adapter_model.bin"))


def test_missing_adapter_path_rejected(tmp_path):
    with pytest.raises(ValueError, match="не найден"):
        preflight.validate_lora_adapter(str(tmp_path / "nope"))


def test_adapter_path_must_be_a_directory(tmp_path):
    file_path = tmp_path / "adapter.bin"
    file_path.write_bytes(b"\0")
    with pytest.raises(ValueError, match="не найден"):
        preflight.validate_lora_adapter(str(file_path))


def test_empty_directory_is_not_an_adapter(tmp_path):
    with pytest.raises(ValueError, match="adapter_config.json"):
        preflight.validate_lora_adapter(str(tmp_path))


def test_adapter_config_must_be_valid_json(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{not json")
    with pytest.raises(ValueError, match="Не удалось прочитать"):
        preflight.validate_lora_adapter(str(tmp_path))


def test_adapter_must_be_lora(tmp_path):
    with pytest.raises(ValueError, match="peft_type=LORA"):
        preflight.validate_lora_adapter(make_adapter(tmp_path, peft_type="PREFIX_TUNING"))


def test_adapter_without_weights_rejected(tmp_path):
    with pytest.raises(ValueError, match="нет весов адаптера"):
        preflight.validate_lora_adapter(make_adapter(tmp_path, weights=None))


def test_load_fails_before_any_model_work_when_adapter_is_bad(script, mocked_loading, tmp_path):
    lora_path = str(tmp_path / "missing")

    with pytest.raises(ValueError):
        script.load_model_and_tokenizer(BASE_MODEL, lora_path)

    assert mocked_loading["model"].calls == []
    assert mocked_loading["tokenizer"].calls == []
    assert mocked_loading["peft"].calls == []


def test_load_without_lora_is_baseline(script, mocked_loading):
    script.load_model_and_tokenizer(BASE_MODEL)
    assert mocked_loading["peft"].calls == []


def test_load_with_valid_adapter_loads_and_merges_it(script, mocked_loading, tmp_path):
    adapter = make_adapter(tmp_path / "adapter")

    result = script.load_model_and_tokenizer(BASE_MODEL, adapter)

    (args, _kwargs), = mocked_loading["peft"].calls
    assert args[1] == adapter
    assert result[0].merged is True  # вернулась модель после merge_and_unload


def test_main_with_missing_adapter_exits_nonzero_without_loading(script, tmp_path, monkeypatch, capsys):
    def must_not_load(*_args, **_kwargs):
        raise AssertionError("модель не должна загружаться при некорректном --lora_model")

    monkeypatch.setattr(script, "load_model_and_tokenizer", must_not_load)
    missing = str(tmp_path / "no_such_adapter")
    argv = ["--base_model", BASE_MODEL, "--lora_model", missing]
    if script is generate_once:
        argv += ["--prompt", "Привет"]

    with pytest.raises(SystemExit) as excinfo:
        script.main(argv)

    assert excinfo.value.code not in (0, None)
    captured = capsys.readouterr()
    assert missing in captured.err
    assert "LoRA:" not in captured.out  # ни слова о применённом адаптере


def test_generate_once_baseline_output_does_not_claim_lora(monkeypatch, capsys):
    monkeypatch.setattr(generate_once, "load_model_and_tokenizer", lambda *_a: (FakeModel(), FakeTokenizer(), "cpu"))
    monkeypatch.setattr(generate_once, "generate_once", lambda *_a: "ответ")

    generate_once.main(["--base_model", BASE_MODEL, "--prompt", "Привет"])

    out = capsys.readouterr().out
    assert "LoRA: не используется" in out
    assert "загружен" not in out


def test_generate_once_reports_lora_when_adapter_is_used(monkeypatch, capsys, tmp_path):
    adapter = make_adapter(tmp_path / "adapter")
    monkeypatch.setattr(generate_once, "load_model_and_tokenizer", lambda *_a: (FakeModel(), FakeTokenizer(), "cpu"))
    monkeypatch.setattr(generate_once, "generate_once", lambda *_a: "ответ")

    generate_once.main(["--base_model", BASE_MODEL, "--lora_model", adapter, "--prompt", "Привет"])

    assert f"LoRA: {adapter}" in capsys.readouterr().out


# --- формат промпта generate_once -----------------------------------------

def test_prompt_format_is_unchanged():
    assert generate_once.build_prompt("Как дела?") == "### Instruction:\nКак дела?\n\n### Response:\n"


def test_prompt_is_a_prefix_of_the_training_text():
    question = "Зачем малому бизнесу чат-бот?"
    training_text = train.format_example({"instruction": question, "output": "Ответ."})
    assert training_text.startswith(generate_once.build_prompt(question))


@pytest.mark.parametrize("extra, expected_seed", [([], 42), (["--seed", "7"], 7)])
def test_main_passes_prompt_and_seed_to_generation(monkeypatch, extra, expected_seed):
    seen = {}

    def fake_generate(_model, _tokenizer, _device, formatted_prompt, *_args):
        seen["prompt"] = formatted_prompt
        seen["seed"] = _args[-1]
        return "ответ"

    monkeypatch.setattr(generate_once, "load_model_and_tokenizer", lambda *_a: (FakeModel(), FakeTokenizer(), "cpu"))
    monkeypatch.setattr(generate_once, "generate_once", fake_generate)

    generate_once.main(["--base_model", BASE_MODEL, "--prompt", "Вопрос?"] + extra)

    assert seen["prompt"] == "### Instruction:\nВопрос?\n\n### Response:\n"
    assert seen["seed"] == expected_seed


def test_generate_once_seeds_immediately_before_sampling(monkeypatch):
    events = []

    class SamplingTokenizer:
        eos_token_id = 1

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[4, 5]]), "attention_mask": torch.tensor([[1, 1]])}

        @staticmethod
        def decode(*_args, **_kwargs):
            return "ответ"

    class SamplingModel:
        device = torch.device("cpu")

        def generate(self, input_ids, **_kwargs):
            events.append("generate")
            return torch.cat([input_ids, torch.tensor([[6]])], dim=1)

    monkeypatch.setattr(generate_once, "set_seed", lambda seed: events.append(f"seed:{seed}"))

    answer = generate_once.generate_once(SamplingModel(), SamplingTokenizer(), "cpu", "prompt", 1, 0.2, 0.85, 7)

    assert answer == "ответ"
    assert events == ["seed:7", "generate"]


# --- проверка аргументов генерации ----------------------------------------

def test_valid_generation_args_pass():
    preflight.validate_generation_args("max_new_tokens", 1, 0.1, 1.0, 0)
    preflight.validate_generation_args("max_length", 512, 0.7, 0.9, 50)
    preflight.validate_generation_args("max_length", 512, 0.7)  # top_p/top_k необязательны


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(length=0), "max_new_tokens"),
        (dict(length=-5), "max_new_tokens"),
        (dict(length=1.5), "max_new_tokens"),
        (dict(temperature=0), "temperature"),
        (dict(temperature=-0.1), "temperature"),
        (dict(temperature=float("nan")), "temperature"),
        (dict(temperature=float("inf")), "temperature"),
        (dict(top_p=0), "top_p"),
        (dict(top_p=1.01), "top_p"),
        (dict(top_p=-0.5), "top_p"),
        (dict(top_p=float("nan")), "top_p"),
        (dict(top_k=-1), "top_k"),
        (dict(top_k=2.5), "top_k"),
    ],
)
def test_invalid_generation_args_rejected(kwargs, message):
    params = dict(length=10, temperature=0.5, top_p=0.9, top_k=50)
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        preflight.validate_generation_args("max_new_tokens", params["length"], params["temperature"],
                                           params["top_p"], params["top_k"])


def test_generate_once_main_rejects_bad_args_before_loading(monkeypatch, capsys):
    def must_not_load(*_args, **_kwargs):
        raise AssertionError("модель не должна загружаться при некорректных аргументах")

    monkeypatch.setattr(generate_once, "load_model_and_tokenizer", must_not_load)

    with pytest.raises(SystemExit) as excinfo:
        generate_once.main(["--base_model", BASE_MODEL, "--prompt", "Привет", "--max_new_tokens", "0"])

    assert excinfo.value.code == 2
    assert "error" in capsys.readouterr().err


def test_chat_main_rejects_bad_args_before_loading(monkeypatch):
    def must_not_load(*_args, **_kwargs):
        raise AssertionError("модель не должна загружаться при некорректных аргументах")

    monkeypatch.setattr(chat, "load_model_and_tokenizer", must_not_load)

    with pytest.raises(SystemExit) as excinfo:
        chat.main(["--base_model", BASE_MODEL, "--max_length", "0"])

    assert excinfo.value.code == 2


def test_chat_generate_response_validates_top_k_and_top_p():
    # проверка идёт до обращения к модели, поэтому модель/токенизатор не нужны
    with pytest.raises(ValueError, match="top_k"):
        chat.generate_response(None, None, "привет", top_k=-1)
    with pytest.raises(ValueError, match="top_p"):
        chat.generate_response(None, None, "привет", top_p=1.5)
