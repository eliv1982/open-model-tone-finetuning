import pytest
import torch

import train

VALID_ARGS = dict(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=512,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    save_steps=500,
    logging_steps=10,
    warmup_steps=100,
    seed=42,
)


def validate(**overrides):
    train.validate_train_args(**{**VALID_ARGS, **overrides})


class TinyTokenizer:
    eos_token_id = 99
    pad_token_id = 99

    @staticmethod
    def encode(text, add_special_tokens=False):
        return list(range(1, len(text.split()) + 1))

    def __call__(self, texts, max_length, **_kwargs):
        return {"input_ids": [self.encode(text)[:max_length] for text in texts]}

    def pad(self, features, padding, return_tensors):
        assert padding is True and return_tensors == "pt"
        longest = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            missing = longest - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * missing)
            attention_mask.append(feature["attention_mask"] + [0] * missing)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


# --- проверка аргументов обучения -----------------------------------------

def test_valid_train_args_pass():
    validate()


def test_train_args_boundaries_that_are_allowed():
    validate(warmup_steps=0)
    validate(lora_dropout=0)
    validate(lora_dropout=0.999)
    validate(seed=0)
    validate(seed=2**32 - 1)


@pytest.mark.parametrize(
    "name, bad_value",
    [
        ("num_train_epochs", 0),
        ("num_train_epochs", -1),
        ("per_device_train_batch_size", 0),
        ("per_device_train_batch_size", 1.5),
        ("gradient_accumulation_steps", 0),
        ("learning_rate", 0),
        ("learning_rate", -1e-4),
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("max_length", 0),
        ("lora_r", 0),
        ("lora_alpha", 0),
        ("lora_alpha", -8),
        ("lora_dropout", -0.01),
        ("lora_dropout", 1),
        ("lora_dropout", 1.5),
        ("save_steps", 0),
        ("logging_steps", 0),
        ("warmup_steps", -1),
        ("warmup_steps", 2.5),
        ("seed", -1),
        ("seed", 2**32),
    ],
)
def test_invalid_train_arg_rejected(name, bad_value):
    with pytest.raises(train.InvalidArguments, match=name):
        validate(**{name: bad_value})


def test_train_validates_before_any_heavy_work(monkeypatch):
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("тяжёлый шаг не должен выполняться при некорректных аргументах")

    for name in ("print_system_info", "load_dataset_from_file", "load_model_and_tokenizer"):
        monkeypatch.setattr(train, name, must_not_run)

    with pytest.raises(train.InvalidArguments, match="num_train_epochs"):
        train.train("some/model", "dataset.json", num_train_epochs=0)


# --- CLI: seed и ошибки аргументов ----------------------------------------

REQUIRED_CLI = ["--model_name", "some/model", "--dataset_path", "dataset.json"]


@pytest.mark.parametrize("extra, expected", [([], 42), (["--seed", "7"], 7)])
def test_cli_seed_default_and_override(monkeypatch, extra, expected):
    seen = {}
    monkeypatch.setattr(train, "train", lambda **kwargs: seen.update(kwargs))

    train.main(REQUIRED_CLI + extra)

    assert seen["seed"] == expected


def test_cli_reports_invalid_argument_and_exits_2(monkeypatch, capsys):
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("тяжёлый шаг не должен выполняться при некорректных аргументах")

    monkeypatch.setattr(train, "load_dataset_from_file", must_not_run)
    monkeypatch.setattr(train, "load_model_and_tokenizer", must_not_run)

    with pytest.raises(SystemExit) as excinfo:
        train.main(REQUIRED_CLI + ["--lora_dropout", "1.0"])

    assert excinfo.value.code == 2
    assert "lora_dropout" in capsys.readouterr().err


# --- EOS и динамический padding --------------------------------------------

def test_eos_stays_trainable_while_padding_is_masked():
    tokenizer = TinyTokenizer()
    dataset = train.preprocess_dataset([{"text": "a b"}, {"text": "a b c d"}], tokenizer, max_length=6)
    batch = train.CausalLMCollator(tokenizer)([dataset[0], dataset[1]])

    assert dataset[0]["input_ids"][-1] == tokenizer.eos_token_id
    assert batch["labels"][0].tolist() == [1, 2, tokenizer.eos_token_id, -100, -100]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]


@pytest.mark.parametrize("max_length, expected", [(4, [1, 2, 3, 99]), (1, [99])])
def test_truncation_reserves_room_for_eos(max_length, expected):
    (example,) = train.preprocess_dataset([{"text": "a b c d e"}], TinyTokenizer(), max_length=max_length)

    assert example["input_ids"] == expected
    assert len(example["input_ids"]) <= max_length


# --- количество шагов: как в Trainer --------------------------------------

@pytest.mark.parametrize(
    "examples, batch, accum, epochs, expected",
    [
        (100, 1, 1, 3, (100, 300)),  # основной сценарий README
        (100, 4, 4, 3, (7, 21)),     # 25 батчей -> ceil(25 / 4) = 7 шагов, а не 6
        (10, 4, 4, 3, (1, 3)),       # маленький датасет: не 0 шагов
        (3, 8, 1, 2, (1, 2)),        # батч больше датасета
        (5, 2, 1, 1, (3, 3)),        # неполный последний батч
        (100, 1, 1, 1, (100, 100)),
    ],
)
def test_compute_training_steps_follows_trainer_semantics(examples, batch, accum, epochs, expected):
    assert train.compute_training_steps(examples, batch, accum, epochs) == expected


def test_compute_training_steps_never_reports_zero_for_small_dataset():
    steps_per_epoch, total = train.compute_training_steps(1, 16, 16, 1)
    assert steps_per_epoch >= 1 and total >= 1


def test_last_logged_loss_skips_the_final_summary_entry():
    history = [
        {"loss": 2.5, "step": 1},
        {"loss": 1.25, "step": 10},
        {"train_runtime": 12.0, "train_loss": 1.8},  # сводка в конце обучения: без ключа 'loss'
    ]
    assert train.last_logged_loss(history) == 1.25
    assert train.last_logged_loss([]) == "N/A"
    assert train.last_logged_loss(None) == "N/A"


def test_callback_epoch_header_counts_from_one(capsys):
    from types import SimpleNamespace

    callback = train.DetailedLoggingCallback(steps_per_epoch=17)
    args = SimpleNamespace(num_train_epochs=2)
    state = SimpleNamespace(epoch=0.0)  # на начало эпохи Trainer хранит число ЗАВЕРШЁННЫХ эпох

    callback.on_epoch_begin(args, state, None)
    state.epoch = 1.0
    callback.on_epoch_begin(args, state, None)

    out = capsys.readouterr().out
    assert "Эпоха 1/2 | Шагов: 17" in out
    assert "Эпоха 2/2 | Шагов: 17" in out
