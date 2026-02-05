<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

This document describes datasets that AIPerf can use to generate stimulus. Additional support is under development, so check back often.

## Synthetic Data Generation

AIPerf can synthetically generate multimodal data for benchmarking:

<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="width:20%; text-align: left;">Data Type</th>
      <th style="width:10%; text-align: center;">Support</th>
      <th style="width:70%; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Synthetic Text</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated text prompts from Shakespeare corpus</td>
    </tr>
    <tr>
      <td><strong>Synthetic Audio</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated audio samples (WAV, MP3 formats)</td>
    </tr>
    <tr>
      <td><strong>Synthetic Images</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated image samples (PNG, JPEG formats)</td>
    </tr>
    <tr>
      <td><strong>Synthetic Video</strong></td>
      <td style="text-align: center;">✅</td>
      <td>Synthetically generated video samples (MP4, WebM formats). Requires FFmpeg.</td>
    </tr>
  </tbody>
</table>

## Custom Dataset Types

Use `--input-file` with `--custom-dataset-type` to load custom datasets. The dataset type can often be auto-detected from the file format.

<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="width:20%; text-align: left;">Dataset Type</th>
      <th style="width:10%; text-align: center;">Support</th>
      <th style="width:70%; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>single_turn</strong></td>
      <td style="text-align: center;">✅</td>
      <td><code>--input-file your_file.jsonl --custom-dataset-type single_turn</code><br/>Single-turn data with multimodal support and client-side batching</td>
    </tr>
    <tr>
      <td><strong>multi_turn</strong></td>
      <td style="text-align: center;">✅</td>
      <td><code>--input-file your_file.jsonl --custom-dataset-type multi_turn</code><br/>Multi-turn conversations with session support, delays, and timestamps</td>
    </tr>
    <tr>
      <td><strong>mooncake_trace</strong></td>
      <td style="text-align: center;">✅</td>
      <td><a href="benchmark_modes/trace_replay.md"><code>--input-file your_trace_file.jsonl --custom-dataset-type mooncake_trace</code></a><br/>Mooncake trace files with timestamps and token lengths for trace replay</td>
    </tr>
    <tr>
      <td><strong>random_pool</strong></td>
      <td style="text-align: center;">✅</td>
      <td><code>--input-file your_dir/ --custom-dataset-type random_pool</code><br/>Directory of files for random sampling. Each file becomes a separate pool.</td>
    </tr>
  </tbody>
</table>

## Public Datasets

Pre-configured public datasets that AIPerf automatically downloads and processes:

<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="width:20%; text-align: left;">Dataset</th>
      <th style="width:10%; text-align: center;">Support</th>
      <th style="width:70%; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>ShareGPT</strong></td>
      <td style="text-align: center;">✅</td>
      <td><code>--public-dataset sharegpt</code><br/>Multi-turn conversations from <a href="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json">HuggingFace</a></td>
    </tr>
  </tbody>
</table>

