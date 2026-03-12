# In-Engine Transport: Credit Fast-Path Architecture

## Overview

When AIPerf benchmarks an LLM engine in-process (vLLM, SGLang, TRT-LLM), the credit
system bypasses ZMQ entirely. Credits flow from the TimingManager directly to the Worker
via Python method calls — zero serialization, zero IPC, zero overhead.

## Standard Mode vs In-Engine Mode

### Standard Mode (HTTP transports)
```
Process 1: SystemController
Process 2: TimingManager ──[ZMQ ROUTER]──> Process N: Worker(s) ──[HTTP]──> LLM Server
Process 3: DatasetManager                       │
Process 4: WorkerManager                        ▼
Process 5: RecordProcessor(s) <──[ZMQ PUSH]─────┘
Process 6: RecordsManager
```

Credits traverse ZMQ ROUTER/DEALER sockets. Workers run in separate processes.
StickyCreditRouter handles load balancing across N workers.

### In-Engine Mode (direct engine API)
```
Process 1: SystemController
Process 2: TimingManager
             ├── InProcessCreditRouter ──[method call]──> Worker ──[Python API]──> Engine
             │         ▲                                    │
             │         └──[callback]── CreditReturn ────────┘
             │                                              │
Process 3: DatasetManager                                   ▼
Process 4: RecordProcessor(s) <──────────[ZMQ PUSH]─────────┘
Process 5: RecordsManager
```

Credits are delivered via `worker.receive_credit(credit)` — a direct Python method call.
Returns flow back via async callbacks. The Worker lives inside TimingManager's process.

## Key Components

### InProcessCreditRouter
**File**: `src/aiperf/credit/in_process_credit_router.py`

Implements the existing `CreditRouterProtocol` — the same interface that `StickyCreditRouter`
implements. The `CreditIssuer` and `PhaseOrchestrator` don't know which router they're talking
to. They just call `send_credit()`, `set_return_callback()`, etc.

```
CreditRouterProtocol
├── StickyCreditRouter    — ZMQ, N workers, load balancing, sticky sessions
└── InProcessCreditRouter — Direct calls, 1 worker, zero overhead
```

**Responsibilities**:
- Holds a reference to the Worker instance
- `send_credit(credit)` → calls `worker.receive_credit(credit)` directly
- `cancel_all_credits()` → calls `worker.cancel_all_credits()` directly
- Wires CreditReturn/FirstToken callbacks from Worker back to PhaseOrchestrator

### Worker Fast-Path Methods
**File**: `src/aiperf/workers/worker.py`

The Worker gained new public methods for in-process credit delivery:

| Method | Purpose |
|--------|---------|
| `receive_credit(credit)` | Accept a credit without ZMQ deserialization |
| `cancel_credits(credit_ids)` | Cancel in-flight credits directly |
| `set_credit_return_callback(cb)` | Route CreditReturn via callback instead of ZMQ |
| `set_first_token_callback(cb)` | Route FirstToken via callback instead of ZMQ |

Internally, `_send_credit_return()` and `_send_first_token()` helpers check for callbacks:
- Callback set? → direct async call (in-process mode)
- No callback? → `credit_dealer_client.send()` via ZMQ (standard mode)

This means the Worker works identically in both modes. The mode is determined by
whether callbacks are wired, not by any flag.

### TimingManager Auto-Detection
**File**: `src/aiperf/timing/manager.py`

TimingManager detects in-engine mode from the endpoint URL scheme:

```python
@staticmethod
def _is_in_engine_mode(user_config: UserConfig) -> bool:
    in_engine_schemes = ("vllm://", "sglang://", "trtllm://")
    return any(url.startswith(in_engine_schemes) for url in user_config.endpoint.urls)
```

When in-engine:
1. Creates `InProcessCreditRouter` (not `StickyCreditRouter`)
2. Creates a Worker in-process via `_create_in_engine_worker()`
3. Attaches Worker to router (`router.attach_worker(worker)`)
4. Attaches Worker to lifecycle (`self.attach_child_lifecycle(worker)`)

When standard HTTP:
- Creates `StickyCreditRouter` as before (no changes)

### InProcessServiceManager
**File**: `src/aiperf/controller/in_process_service_manager.py`

Extends `MultiProcessServiceManager` with one override: blocks Worker spawning.
Since the Worker lives inside TimingManager's process, the service manager must not
spawn separate Worker processes.

```python
class InProcessServiceManager(MultiProcessServiceManager):
    async def run_service(self, service_type, num_replicas=1):
        if service_type == ServiceType.WORKER:
            return  # Worker is owned by InProcessCreditRouter
        await super().run_service(service_type, num_replicas)
```

Registered in `plugins.yaml` as `service_manager.in_engine`.

## Credit Lifecycle (In-Engine)

```
1. PhaseOrchestrator registers callbacks on InProcessCreditRouter
2. CreditIssuer acquires session/prefill slots
3. CreditIssuer calls router.send_credit(credit)
4. InProcessCreditRouter calls worker.receive_credit(credit)
5. Worker._schedule_credit_drop_task(credit)  →  creates async task
6. Task runs _process_credit():
   a. Fetches conversation from DatasetManager (ZMQ)
   b. Builds request via InferenceClient
   c. Calls transport._generate() → engine.generate_async()
   d. Sends FirstToken via callback → PhaseOrchestrator releases prefill slot
   e. Collects response, builds RequestRecord
7. Worker sends CreditReturn via callback → PhaseOrchestrator releases session slot
8. Worker pushes InferenceResults via ZMQ PUSH → RecordProcessor (still ZMQ)
```

Steps 4, 6d, and 7 are the fast-path — direct method calls instead of ZMQ.
Step 8 is still ZMQ because RecordProcessor runs in a separate process (by design).

## What Changes vs What Stays

### Changed
| Component | Change |
|-----------|--------|
| `TimingManager.__init__` | Branches on URL scheme → InProcessCreditRouter or StickyCreditRouter |
| `Worker` | New public methods for in-process credit delivery + callback routing |
| `InProcessCreditRouter` | New class implementing CreditRouterProtocol |
| `InProcessServiceManager` | New class blocking Worker spawning |
| `plugins.yaml` | New `in_engine` service manager entry |

### Unchanged (100%)
| Component | Why |
|-----------|-----|
| CreditIssuer | Depends on CreditRouterProtocol, not concrete class |
| PhaseOrchestrator | Depends on CreditRouterProtocol, not concrete class |
| StickyCreditRouter | Not modified, still used for HTTP mode |
| CreditCallbackHandler | Same callbacks, doesn't know about transport |
| InferenceClient | Same interface, called identically |
| BaseInEngineTransport | Same _generate() path |
| vLLM/SGLang/TRT-LLM transports | No changes |
| RecordProcessor | Still receives via ZMQ PUSH |
| RecordsManager | Still aggregates normally |
| DatasetManager | Still provides conversations via ZMQ |
| All existing tests | No behavioral changes |


## Design Principles

1. **Protocol-first**: `CreditRouterProtocol` was designed for exactly this — "Enables alternative routing strategies"
2. **Zero-flag design**: No `in_process=True` flags. Mode is determined by which protocol implementation is injected.
3. **Backward compatible**: HTTP mode is completely untouched. InProcessCreditRouter is additive.
4. **Minimal blast radius**: 2 new files, 2 modified files. Everything else is unchanged.
5. **Same event loop**: Worker and TimingManager share an event loop. Credits are function calls. No queues, no background tasks, no serialization.
