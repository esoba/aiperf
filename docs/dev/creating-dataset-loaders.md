<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Creating Dataset Loaders

This guide walks you through creating a custom dataset loader plugin for AIPerf. By the end, you'll have a working loader that reads data from a custom file format and integrates with the AIPerf plugin system.

## What You Will Build

We'll create a hypothetical **CSV dataset loader** that reads prompts from CSV files. This demonstrates all the concepts needed for more complex loaders.

## Architecture Overview

### Class Hierarchy

```
BaseDatasetLoader (ABC)          # Root: model selection, output token sampling, turn finalization, context prompts
├── BaseFileLoader               # File-based: two-stage pipeline (parse → convert)
│   ├── SingleTurnDatasetLoader  #   + MediaConversionMixin
│   ├── MultiTurnDatasetLoader   #   + MediaConversionMixin
│   ├── RandomPoolDatasetLoader  #   + MediaConversionMixin
│   ├── MooncakeTraceDatasetLoader
│   └── ShareGPTLoader
└── BaseSyntheticLoader          # Synthetic: generator management, ISL/OSL pairing
    ├── SyntheticMultiModalLoader
    └── SyntheticRankingsLoader
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **`BaseDatasetLoader`** | Root ABC. Manages model selection, output token sampling, turn finalization, and context prompt injection. |
| **`BaseFileLoader`** | For file-based loaders. Implements a two-stage pipeline: `parse_and_validate()` then `convert_to_conversations()`. Finalization is automatic. |
| **`BaseSyntheticLoader`** | For loaders that generate data programmatically. Provides `_generate_text_payloads()`, `_generate_image_payloads()`, etc. |
| **`MediaConversionMixin`** | Optional mixin that converts modality fields (`text`/`texts`, `image`/`images`, etc.) into `Media` objects. |
| **`DatasetLoaderProtocol`** | The protocol all loaders satisfy: `can_load()`, `get_preferred_sampling_strategy()`, `load()`. |

### Data Flow

```
Input File → parse_and_validate() → Typed Model Objects
                                         ↓
                              convert_to_conversations()
                                         ↓
                              List[Conversation] (raw)
                                         ↓
                              _finalize_turn() per turn     ← model name, max_tokens
                              _finalize_conversations()     ← context prompts
                                         ↓
                              List[Conversation] (finalized) → DatasetManager
```

For `BaseFileLoader` subclasses, the `load()` method runs this pipeline automatically. You only implement the first two stages.

## Step-by-Step: Creating a File-Based Loader

### Step 1: Define Your Data Model

Create a Pydantic model representing one line of your input file. Every field must have `Field(description=...)`.

```python
# my_aiperf_plugins/loaders/models.py
from pydantic import Field
from aiperf.common.models import AIPerfBaseModel

class CsvEntry(AIPerfBaseModel):
    """Schema for a single CSV-derived entry."""

    prompt: str = Field(..., description="The prompt text to send to the model")
    expected_tokens: int | None = Field(
        None, description="Expected output token count"
    )
```

### Step 2: Implement the Loader

Subclass `BaseFileLoader` and implement the three required abstract methods (plus `can_load_file` if auto-detection is desired).

```python
# my_aiperf_plugins/loaders/csv_loader.py
import csv
from pathlib import Path
from typing import Any

from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.file.base import BaseFileLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

from .models import CsvEntry


class CsvDatasetLoader(BaseFileLoader):
    """Dataset loader that reads prompts from a CSV file.

    Expected CSV format:
        prompt,expected_tokens
        "What is deep learning?",128
        "Explain transformers",256
    """

    @classmethod
    def can_load_file(
        cls, data: dict[str, Any], filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data.

        CSV files are not JSONL, so we cannot auto-detect from data content.
        Users must specify --dataset-type csv explicitly.
        """
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Sequential sampling preserves CSV row order."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def parse_and_validate(self) -> dict[str, list[CsvEntry]]:
        """Parse the CSV file into validated CsvEntry objects.

        Returns:
            Dict mapping session_id to list of CsvEntry objects.
        """
        data: dict[str, list[CsvEntry]] = {}

        with open(self.filename, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = CsvEntry(
                    prompt=row["prompt"],
                    expected_tokens=int(row["expected_tokens"])
                    if row.get("expected_tokens")
                    else None,
                )
                session_id = self.session_id_generator.next()
                data[session_id] = [entry]

        return data

    def convert_to_conversations(
        self, data: dict[str, list[CsvEntry]]
    ) -> list[Conversation]:
        """Convert parsed CSV entries to Conversation objects.

        Args:
            data: Dict mapping session_id to list of CsvEntry objects.

        Returns:
            List of Conversation objects.
        """
        conversations = []
        for session_id, entries in data.items():
            conversation = Conversation(session_id=session_id)
            for entry in entries:
                turn = Turn(
                    texts=[Text(name="text", contents=[entry.prompt])],
                    max_tokens=entry.expected_tokens,
                )
                conversation.turns.append(turn)
            conversations.append(conversation)
        return conversations
```

### Step 3: Optional Multi-Modal Support

If your loader handles images, audio, or video, mix in `MediaConversionMixin`:

```python
from aiperf.dataset.loader.mixins import MediaConversionMixin

class MyMultiModalLoader(BaseFileLoader, MediaConversionMixin):
    def convert_to_conversations(self, data):
        # Use self.convert_to_media_objects() to handle modality fields
        for session_id, entries in data.items():
            for entry in entries:
                media = self.convert_to_media_objects(entry)
                turn = Turn(
                    texts=media[MediaType.TEXT],
                    images=media[MediaType.IMAGE],
                    # ...
                )
```

### Step 4: Register in plugins.yaml

```yaml
# my_aiperf_plugins/plugins.yaml
schema_version: "1.0"

dataset_loader:
  csv:
    class: my_aiperf_plugins.loaders.csv_loader:CsvDatasetLoader
    description: |
      CSV dataset loader for loading prompts from CSV files with optional
      expected output token counts.
```

### Step 5: Set Up pyproject.toml

```toml
[project]
name = "my-aiperf-plugins"
version = "0.1.0"
dependencies = ["aiperf"]

[project.entry-points."aiperf.plugins"]
my-aiperf-plugins = "my_aiperf_plugins:plugins.yaml"
```

### Step 6: Install and Verify

```bash
uv pip install -e .
aiperf plugins dataset_loader csv
```

You should see your loader listed with its class path and description.

### Step 7: Use Your Loader

```bash
aiperf --input-file data.csv --dataset-type csv --endpoint-type chat --model my-model
```

## Creating a Synthetic Loader

Synthetic loaders extend `BaseSyntheticLoader` and generate data programmatically instead of reading from files.

```python
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.synthetic.base import BaseSyntheticLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class MySyntheticLoader(BaseSyntheticLoader):
    """Generates synthetic data for my custom endpoint."""

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SHUFFLE

    def load(self) -> list[Conversation]:
        conversations = []
        for _ in range(self.config.input.conversation.num_dataset_entries):
            conversation = Conversation(session_id=self.session_id_generator.next())
            turn = Turn()

            # Use built-in generators
            if self.include_prompt:
                turn.texts.append(self._generate_text_payloads(turn, is_first=True))
            if self.include_image:
                turn.images.append(self._generate_image_payloads())

            self._finalize_turn(turn)
            conversation.turns.append(turn)
            conversations.append(conversation)

        self._finalize_conversations(conversations)
        return conversations
```

`BaseSyntheticLoader` provides these helpers:

| Helper | Returns | Description |
|--------|---------|-------------|
| `_generate_text_payloads(turn, is_first)` | `Text` | Generates text with ISL/OSL distribution and prefix prompts |
| `_generate_image_payloads()` | `Image` | Generates synthetic images based on config |
| `_generate_audio_payloads()` | `Audio` | Generates synthetic audio based on config |
| `_generate_video_payloads()` | `Video` | Generates synthetic video based on config |
| `include_prompt` | `bool` | Whether text generation is enabled |
| `include_image` | `bool` | Whether image generation is enabled |
| `include_audio` | `bool` | Whether audio generation is enabled |
| `include_video` | `bool` | Whether video generation is enabled |

**Important:** Synthetic loaders must call `_finalize_turn()` on each turn and `_finalize_conversations()` on the full list. `BaseFileLoader` does this automatically, but synthetic loaders override `load()` directly.

## The `can_load()` Contract

The `can_load()` method powers auto-detection. Best practices:

1. **Return `True` only when confident.** If your format is ambiguous with another loader, return `False` and require explicit `--dataset-type`.
2. **Use Pydantic validation.** Validate the sample data against your model:
   ```python
   @classmethod
   def can_load_file(cls, data, filename=None):
       try:
           MyModel.model_validate(data)
           return True
       except ValidationError:
           return False
   ```
3. **Check `type` field first.** If your format uses a `type` discriminator, check it before structural validation.
4. **For directories**, override `can_load_directory()` (only `random_pool` does this by default).
5. **Synthetic loaders** always return `False` from `can_load()` since they don't load from files.

## Understanding Turn Finalization

`BaseDatasetLoader` provides finalization that runs automatically for `BaseFileLoader` subclasses (in the `load()` pipeline) and must be called explicitly in synthetic loaders.

### `_finalize_turn(turn)`

Called per turn. Does three things:

1. **Model selection** -- assigns `turn.model` using the configured model selection strategy
2. **Output token sampling** -- sets `turn.max_tokens` from sequence distribution or output config
3. **Cache cleanup** -- clears cached ISL/OSL values for the turn

### `_finalize_conversations(conversations)`

Called once on the full list. Injects context prompts:

- **Shared system prompt** (`--shared-system-prompt-length`) -- same system message for all conversations
- **User context prompt** (`--user-context-prompt-length`) -- unique per-session context

## Testing Your Loader

### Test Structure

```python
import pytest
from aiperf.common.models import Conversation
from aiperf.plugin.enums import DatasetSamplingStrategy
from my_aiperf_plugins.loaders.csv_loader import CsvDatasetLoader


class TestCsvDatasetLoader:
    """Tests for CsvDatasetLoader."""

    def test_can_load_file_returns_false(self):
        """CSV loader requires explicit --dataset-type."""
        assert CsvDatasetLoader.can_load_file({"prompt": "test"}) is False

    def test_get_preferred_sampling_strategy(self):
        assert (
            CsvDatasetLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    def test_parse_and_validate(self, loader_with_csv_file):
        """Test that CSV rows are parsed into CsvEntry objects."""
        data = loader_with_csv_file.parse_and_validate()
        assert len(data) == 3  # 3 rows in fixture

    def test_convert_to_conversations(self, loader_with_csv_file):
        """Test that parsed data becomes Conversation objects."""
        data = loader_with_csv_file.parse_and_validate()
        conversations = loader_with_csv_file.convert_to_conversations(data)
        assert all(isinstance(c, Conversation) for c in conversations)
        assert all(len(c.turns) == 1 for c in conversations)

    @pytest.mark.parametrize(
        "prompt,expected_tokens",
        [
            ("Hello", None),
            ("Test prompt", 128),
        ],
    )
    def test_entry_fields(self, prompt, expected_tokens):
        """Test individual entry parsing."""
        from my_aiperf_plugins.loaders.models import CsvEntry

        entry = CsvEntry(prompt=prompt, expected_tokens=expected_tokens)
        assert entry.prompt == prompt
        assert entry.expected_tokens == expected_tokens
```

### What to Test

| Area | What to Verify |
|------|---------------|
| `can_load_file` / `can_load` | Returns `True` for valid data, `False` for invalid or ambiguous data |
| `parse_and_validate` | Correct parsing, handles empty lines, validates schema |
| `convert_to_conversations` | Correct `Conversation`/`Turn` structure, modality fields populated |
| `get_preferred_sampling_strategy` | Returns the expected strategy |
| Edge cases | Empty files, malformed lines, missing fields |

## Reference: Methods to Implement

### BaseFileLoader

| Method | Required | Description |
|--------|:--------:|-------------|
| `can_load_file(data, filename)` | No | Auto-detection from first JSONL line (default: `False`) |
| `can_load_directory(path)` | No | Auto-detection from directory path (default: `False`) |
| `get_preferred_sampling_strategy()` | Yes | Default sampling when user doesn't specify |
| `parse_and_validate()` | Yes | Parse file(s) into typed model objects |
| `convert_to_conversations(data)` | Yes | Convert parsed data to `Conversation` objects |

### BaseSyntheticLoader

| Method | Required | Description |
|--------|:--------:|-------------|
| `can_load(data, filename)` | No | Override only if needed (default: `False`) |
| `get_preferred_sampling_strategy()` | Yes | Default sampling when user doesn't specify |
| `load()` | Yes | Generate and return finalized `Conversation` objects |
