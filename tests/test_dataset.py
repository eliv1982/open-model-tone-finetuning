import json
import os

import pytest

import train
from conftest import PROJECT_ROOT


def write_json(tmp_path, content, name="data.json"):
    path = tmp_path / name
    path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return str(path)


def write_text(tmp_path, text, name):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


VALID = {"instruction": "Вопрос?", "output": "Ответ."}


# --- валидные данные -------------------------------------------------------

def test_valid_json_list(tmp_path):
    path = write_json(tmp_path, [VALID, {"instruction": "Ещё?", "output": "Да."}])
    assert train.load_dataset_from_file(path) == [VALID, {"instruction": "Ещё?", "output": "Да."}]


def test_valid_json_object_with_data_key(tmp_path):
    path = write_json(tmp_path, {"version": 1, "data": [VALID]})
    assert train.load_dataset_from_file(path) == [VALID]


def test_repository_dataset_is_valid():
    data = train.load_dataset_from_file(os.path.join(PROJECT_ROOT, "data", "business_tone_dataset.json"))
    assert len(data) == 100
    assert all(set(record) == {"instruction", "output"} for record in data)


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"text": "Просто текст"}, "Просто текст"),
        ({"instruction": "Q", "output": "A"}, "### Instruction:\nQ\n\n### Response:\nA"),
        ({"prompt": "P", "completion": "C"}, "P\n\nC"),
        ({"input": "I", "output": "O"}, "Input: I\nOutput: O"),
    ],
)
def test_supported_schemas_are_loaded_and_formatted(tmp_path, record, expected):
    path = write_json(tmp_path, [record])
    (loaded,) = train.load_dataset_from_file(path)
    assert train.format_example(loaded) == expected


def test_valid_jsonl_skips_blank_lines_and_bom(tmp_path):
    path = tmp_path / "data.jsonl"
    body = json.dumps(VALID, ensure_ascii=False) + "\n\n" + json.dumps({"text": "t"}) + "\n"
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    assert train.load_dataset_from_file(str(path)) == [VALID, {"text": "t"}]


# --- повреждённый / неподдерживаемый JSON ---------------------------------

def test_malformed_json_rejected(tmp_path):
    path = write_text(tmp_path, '[{"instruction": "a", "output": ', "data.json")
    with pytest.raises(ValueError, match="некорректный JSON"):
        train.load_dataset_from_file(path)


@pytest.mark.parametrize("content", ["42", '"just a string"', "null"])
def test_json_top_level_scalar_rejected(tmp_path, content):
    path = write_text(tmp_path, content, "data.json")
    with pytest.raises(ValueError, match="списком"):
        train.load_dataset_from_file(path)


def test_json_object_without_data_key_is_not_guessed(tmp_path):
    # раньше бралось первое значение словаря - теперь это ошибка
    path = write_json(tmp_path, {"examples": [VALID]})
    with pytest.raises(ValueError, match="ключом 'data'"):
        train.load_dataset_from_file(path)


def test_json_data_key_must_hold_a_list(tmp_path):
    path = write_json(tmp_path, {"data": {"instruction": "a", "output": "b"}})
    with pytest.raises(ValueError, match="списком"):
        train.load_dataset_from_file(path)


def test_unsupported_extension_rejected(tmp_path):
    path = write_text(tmp_path, "text", "data.txt")
    with pytest.raises(ValueError, match="json"):
        train.load_dataset_from_file(path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        train.load_dataset_from_file(str(tmp_path / "nope.json"))


# --- JSONL: номер строки ---------------------------------------------------

def test_malformed_jsonl_line_reports_line_number(tmp_path):
    lines = [json.dumps(VALID, ensure_ascii=False), json.dumps(VALID, ensure_ascii=False), "{oops", json.dumps(VALID)]
    path = write_text(tmp_path, "\n".join(lines) + "\n", "data.jsonl")
    with pytest.raises(ValueError, match="строка 3: некорректный JSON"):
        train.load_dataset_from_file(path)


def test_jsonl_line_number_counts_blank_lines(tmp_path):
    text = json.dumps(VALID, ensure_ascii=False) + "\n\n\n[1, 2]\n"
    path = write_text(tmp_path, text, "data.jsonl")
    with pytest.raises(ValueError, match="строка 4: запись должна быть JSON-объектом"):
        train.load_dataset_from_file(path)


# --- пустой датасет --------------------------------------------------------

@pytest.mark.parametrize("content", [[], {"data": []}])
def test_empty_json_dataset_rejected(tmp_path, content):
    path = write_json(tmp_path, content)
    with pytest.raises(ValueError, match="пуст"):
        train.load_dataset_from_file(path)


@pytest.mark.parametrize("text", ["", "\n\n  \n"])
def test_empty_jsonl_dataset_rejected(tmp_path, text):
    path = write_text(tmp_path, text, "data.jsonl")
    with pytest.raises(ValueError, match="пуст"):
        train.load_dataset_from_file(path)


# --- некорректные записи ---------------------------------------------------

@pytest.mark.parametrize(
    "record, message",
    [
        ("просто строка", "должна быть JSON-объектом"),
        ([1, 2], "должна быть JSON-объектом"),
        (None, "должна быть JSON-объектом"),
        ({}, "неподдерживаемая запись"),
        ({"question": "q", "answer": "a"}, "неподдерживаемая запись"),
        ({"instruction": "только вопрос"}, "неподдерживаемая запись"),
        ({"output": "только ответ"}, "неподдерживаемая запись"),
        ({"instruction": 5, "output": "a"}, "поле 'instruction' должно быть непустой строкой"),
        ({"instruction": "q", "output": None}, "поле 'output' должно быть непустой строкой"),
        ({"instruction": "q", "output": ""}, "поле 'output' должно быть непустой строкой"),
        ({"instruction": "   ", "output": "a"}, "поле 'instruction' должно быть непустой строкой"),
        ({"text": ["не", "строка"]}, "поле 'text' должно быть непустой строкой"),
        ({"text": ""}, "поле 'text' должно быть непустой строкой"),
    ],
)
def test_invalid_record_rejected(tmp_path, record, message):
    path = write_json(tmp_path, [VALID, record])
    with pytest.raises(ValueError, match=message) as excinfo:
        train.load_dataset_from_file(path)
    assert "запись 2" in str(excinfo.value)  # валидная запись 1 пропущена, ошибка указывает на запись 2


def test_invalid_jsonl_record_reports_line_number(tmp_path):
    text = json.dumps(VALID, ensure_ascii=False) + "\n" + json.dumps({"instruction": "q", "output": ""}) + "\n"
    path = write_text(tmp_path, text, "data.jsonl")
    with pytest.raises(ValueError, match="строка 2: поле 'output'"):
        train.load_dataset_from_file(path)


def test_format_example_rejects_unsupported_record():
    with pytest.raises(ValueError, match="Неподдерживаемая запись"):
        train.format_example({"foo": "bar"})
