<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compressed Dataset Support

AIPerf can load datasets directly from compressed files and archives, avoiding the need to manually decompress before benchmarking. This works with all custom dataset types (`single_turn`, `multi_turn`, `random_pool`, `mooncake_trace`, `bailian_trace`).

## Supported Formats

| Type | Extensions |
|------|-----------|
| Single-file compression | `.gz`, `.zst`, `.xz` |
| Multi-file archives | `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.zst`, `.tar.xz` |

---

## Single-File Compression

For `.gz`, `.zst`, and `.xz` files, AIPerf decompresses the file and infers the inner filename by stripping the compression extension (e.g., `prompts.jsonl.zst` becomes `prompts.jsonl`).

```bash
# Compress with any supported format
gzip prompts.jsonl       # creates prompts.jsonl.gz
zstd prompts.jsonl       # creates prompts.jsonl.zst
xz prompts.jsonl         # creates prompts.jsonl.xz

# Pass the compressed file directly
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file prompts.jsonl.zst \
    --streaming \
    --url localhost:8000
```

Use `--input-file-subpath` to override the inferred filename if needed:

```bash
aiperf profile \
    --input-file prompts.jsonl.zst \
    --input-file-subpath custom_name.jsonl \
    ...
```

---

## Multi-File Archives

For archives containing multiple files, use `--input-file-subpath` to specify which file to load. This is required for file-based dataset types (`single_turn`, `multi_turn`, `mooncake_trace`, `bailian_trace`).

```bash
# zip
zip dataset.zip data/prompts.jsonl
aiperf profile --input-file dataset.zip --input-file-subpath data/prompts.jsonl ...

# tar.gz
tar czf dataset.tar.gz data/prompts.jsonl
aiperf profile --input-file dataset.tar.gz --input-file-subpath data/prompts.jsonl ...

# tar.zst
tar cf - data/prompts.jsonl | zstd -o dataset.tar.zst
aiperf profile --input-file dataset.tar.zst --input-file-subpath data/prompts.jsonl ...
```

### Archives as Directories

For directory-based dataset types like `random_pool`, omit `--input-file-subpath` to extract the entire archive. Each file in the archive becomes a separate pool:

```bash
zip pools.zip pool_a.jsonl pool_b.jsonl pool_c.jsonl

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --input-file pools.zip \
    --custom-dataset-type random_pool \
    --num-conversations 100 \
    --streaming \
    --url localhost:8000
```

---

## How It Works

1. AIPerf detects the compression format from the file extension
2. The file is decompressed or extracted to a temporary directory
3. The dataset is loaded using the standard loader pipeline
4. The temporary directory is cleaned up after loading completes

Decompression uses chunked I/O to limit memory usage. For `tar.zst` archives, decompression is fully streaming.

---

## CLI Reference

| Option | Description |
|--------|-------------|
| `--input-file` | Path to the compressed file or archive |
| `--input-file-subpath` | Relative path to a file inside an archive, or override for the inferred filename in single-file compression |
| `--custom-dataset-type` | Dataset format (auto-detected if omitted) |

---

## See Also

- [Custom Dataset Guide](custom-dataset.md) - Dataset formats and types
- [Fixed Schedule](fixed-schedule.md) - Timestamp-based replay with compressed trace files
