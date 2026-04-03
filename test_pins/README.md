# PinAssistant Multi-Turn Replay Data Generator

Generates multi-turn JSONL datasets for benchmarking the PinAssistant agentic framework using AIPerf's raw payload replay mode.

## Setup

Requires the `ajc/raw-payload-support` branch:

```bash
git clone git@github.com:ai-dynamo/aiperf.git
cd aiperf
git checkout ajc/raw-payload-support
make first-time-setup
. .venv/bin/activate
```

## Generate Data

```bash
python test_pins/generate_replay_data.py \
    --config test_pins/example_configs/pinassistant.yaml \
    --output-dir /tmp/pinassistant_replay \
    --num-conversations 100 \
    --image-pool-size 300 \
    --seed 42
```

Options:

- `--image-mode base64` (default): generates random PNGs as base64 data URLs
- `--image-mode http --coco-annotations <path>`: samples COCO image URLs
- `--image-pool-size`: must be >= (images per conversation) x (num conversations) since images are sampled without replacement

## Run Benchmark

```bash
aiperf-mock-server &

aiperf profile \
    --input-file /tmp/pinassistant_replay \
    --model test \
    --custom-dataset-type raw_payload \
    --endpoint-type raw \
    --url localhost:8000/v1/chat/completions \
    --concurrency 5
```

## Known Issue: `NotImplementedError` on `ajc/raw-payload-support`

### Reproducing

```bash
aiperf-mock-server &

python test_pins/generate_replay_data.py \
    --config test_pins/example_configs/pinassistant.yaml \
    --output-dir /tmp/pinassistant_sample \
    --num-conversations 2 \
    --image-pool-size 6 \
    --seed 42

aiperf profile \
    --input-file /tmp/pinassistant_sample \
    --model test \
    --custom-dataset-type raw_payload \
    --endpoint-type raw \
    --url localhost:8000/v1/chat/completions \
    --concurrency 2
```

Fails with:

```
NotImplementedError: RawEndpoint does not format payloads.
Use raw_payload or inputs_json dataset types.
```

### Root Cause

During "Configure Profiling", `DatasetManager._generate_inputs_json_file()` calls `format_conversation_payloads()` to produce an `inputs.json` artifact. That function unconditionally calls `endpoint.format_payload(request_info)` at `src/aiperf/dataset/payload_formatting.py:59`. `RawEndpoint.format_payload()` raises `NotImplementedError` by design -- raw payload turns bypass endpoint formatting entirely.

### Fix

In `src/aiperf/dataset/payload_formatting.py`, check for `turn.raw_payload` before calling `endpoint.format_payload()`:

```diff
-            yield conversation.session_id, i, endpoint.format_payload(request_info)
+            if turn.raw_payload is not None:
+                yield conversation.session_id, i, turn.raw_payload
+            else:
+                yield conversation.session_id, i, endpoint.format_payload(request_info)
```

Turns loaded by `RawPayloadDatasetLoader` already carry the complete API request body in `turn.raw_payload`. This is consistent with how `InferenceClient._send_request_to_transport()` handles it at `src/aiperf/workers/inference_client.py:101-104`.
