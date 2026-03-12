# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

from aiperf.common.config.accuracy_config import AccuracyConfig as LegacyAccuracyConfig
from aiperf.common.config.audio_config import AudioConfig, AudioLengthConfig
from aiperf.common.config.conversation_config import (
    ConversationConfig,
    TurnConfig,
    TurnDelayConfig,
)
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.image_config import (
    ImageConfig,
    ImageHeightConfig,
    ImageWidthConfig,
)
from aiperf.common.config.input_config import InputConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.output_config import OutputConfig
from aiperf.common.config.prompt_config import (
    InputTokensConfig,
    OutputTokensConfig,
    PrefixPromptConfig,
    PromptConfig,
)
from aiperf.common.config.rankings_config import (
    RankingsConfig,
    RankingsPassagesConfig,
    RankingsQueryConfig,
)
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.config.synthesis_config import SynthesisConfig
from aiperf.common.config.tokenizer_config import TokenizerConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.config.video_config import VideoConfig
from aiperf.common.config.worker_config import WorkersConfig
from aiperf.common.config.zmq_config import ZMQIPCConfig, ZMQTCPConfig
from aiperf.common.enums import AIPerfLogLevel, ExportLevel, GPUTelemetryMode
from aiperf.config.config import AIPerfConfig
from aiperf.config.dataset import (
    ComposedDataset,
    DatasetConfig,
    FileDataset,
    PublicDataset,
    SyntheticDataset,
)
from aiperf.config.models import IpcCommunicationConfig, TcpCommunicationConfig
from aiperf.config.phases import PhaseConfig, PhaseType
from aiperf.config.types import MeanStddev
from aiperf.plugin.enums import ArrivalPattern, TimingMode


def _omit(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


def _ms(v: MeanStddev | None) -> tuple[float | None, float | None]:
    return (v.mean, v.stddev) if v else (None, None)


def _int(v: float | None) -> int | None:
    return int(v) if v is not None else None


def _ramp(r: object | None) -> float | None:
    return r.duration if r else None  # type: ignore[union-attr]


def _multimodal(ds: DatasetConfig, field: str) -> Any:
    if isinstance(ds, SyntheticDataset):
        return getattr(ds, field)
    if isinstance(ds, ComposedDataset):
        return getattr(ds.augment, field)
    return None


def convert_to_legacy_configs(
    config: AIPerfConfig,
) -> tuple[UserConfig, ServiceConfig]:
    return _build_user_config(config), _build_service(config)


def _build_service(config: AIPerfConfig) -> ServiceConfig:
    comm = config.runtime.communication
    if isinstance(comm, IpcCommunicationConfig):
        zmq: ZMQIPCConfig | ZMQTCPConfig = ZMQIPCConfig(path=Path(comm.path))
    elif isinstance(comm, TcpCommunicationConfig):
        from aiperf.common.config.zmq_config import ZMQTCPProxyConfig

        _proxy = lambda p: ZMQTCPProxyConfig(  # noqa: E731
            host=comm.host, frontend_port=p.frontend_port, backend_port=p.backend_port
        )
        zmq = ZMQTCPConfig.model_construct(
            host=comm.host,
            records_push_pull_port=comm.records_port,
            credit_router_port=comm.credit_router_port,
            event_bus_proxy_config=_proxy(comm.event_bus_proxy),
            dataset_manager_proxy_config=_proxy(comm.dataset_manager_proxy),
            raw_inference_proxy_config=_proxy(comm.raw_inference_proxy),
        )
    else:
        zmq = ZMQIPCConfig()
    log = config.logging.level
    sc = ServiceConfig.model_construct(
        zmq_tcp=zmq if isinstance(zmq, ZMQTCPConfig) else None,
        zmq_ipc=zmq if isinstance(zmq, ZMQIPCConfig) else None,
        workers=WorkersConfig(max=config.runtime.workers),
        log_level=log,
        verbose=log in (AIPerfLogLevel.DEBUG, AIPerfLogLevel.TRACE),
        extra_verbose=log == AIPerfLogLevel.TRACE,
        record_processor_service_count=config.runtime.record_processors,
        ui_type=config.runtime.ui,
    )
    sc.__dict__["_comm_config"] = zmq
    return sc


def _build_user_config(config: AIPerfConfig) -> UserConfig:
    prof = next((p for p in config.load.values() if not p.exclude), None)
    gt, sm = config.gpu_telemetry, config.server_metrics
    if prof and prof.type == PhaseType.FIXED_SCHEDULE:
        tm = TimingMode.FIXED_SCHEDULE
    elif prof and prof.type == PhaseType.USER_CENTRIC:
        tm = TimingMode.USER_CENTRIC_RATE
    else:
        tm = TimingMode.REQUEST_RATE
    ep = config.endpoint
    a = config.artifacts
    tok = config.tokenizer
    if a.raw:
        lvl = ExportLevel.RAW
    elif a.records is not False:
        lvl = ExportLevel.RECORDS
    else:
        lvl = ExportLevel.SUMMARY
    uc = UserConfig.model_construct(
        endpoint=EndpointConfig(
            model_names=config.get_model_names(),
            model_selection_strategy=config.models.strategy,
            custom_endpoint=ep.path,
            type=ep.type,
            streaming=ep.streaming,
            urls=list(ep.urls),
            url_selection_strategy=ep.url_strategy,
            timeout_seconds=ep.timeout,
            api_key=ep.api_key,
            transport=ep.transport,
            use_legacy_max_tokens=ep.use_legacy_max_tokens,
            use_server_token_count=ep.use_server_token_count,
            connection_reuse_strategy=ep.connection_reuse,
            download_video_content=ep.download_video_content,
        ),
        input=_build_input(config, prof),
        output=OutputConfig(
            artifact_directory=a.dir,
            profile_export_prefix=Path(a.prefix) if a.prefix != "profile" else None,
            export_level=lvl,
            slice_duration=a.slice_duration,
            export_http_trace=a.trace,
            show_trace_timing=a.show_trace_timing,
        ),
        tokenizer=TokenizerConfig()
        if not tok
        else TokenizerConfig(
            name=tok.name,
            revision=tok.revision,
            trust_remote_code=tok.trust_remote_code,
        ),
        loadgen=_build_loadgen(config, prof),
        cli_command=a.cli_command,
        benchmark_id=a.benchmark_id,
        gpu_telemetry=list(gt.urls) if gt.enabled and gt.urls else None,
        no_gpu_telemetry=not gt.enabled,
        server_metrics=list(sm.urls) if sm.enabled and sm.urls else None,
        no_server_metrics=not sm.enabled,
        server_metrics_formats=sm.formats,
        accuracy=_build_accuracy(config),
    )
    uc.__dict__ |= {
        "_timing_mode": tm,
        "_gpu_telemetry_mode": gt.mode if gt.enabled else GPUTelemetryMode.SUMMARY,
        "_gpu_telemetry_collector_type": "dcgm",
        "_gpu_telemetry_urls": list(gt.urls) if gt.enabled else [],
        "_gpu_telemetry_metrics_file": gt.metrics_file if gt.enabled else None,
        "_server_metrics_urls": list(sm.urls) if sm.enabled else [],
    }
    return uc


def _build_input(config: AIPerfConfig, prof: PhaseConfig | None):
    ds = config.get_default_dataset()
    fp = pub = cdt = samp = None
    fixed = auto_off = False
    start_off = end_off = None
    if isinstance(ds, FileDataset):
        fp = str(ds.path)
        cdt = str(ds.format) if ds.format else None
        samp = ds.sampling
    elif isinstance(ds, PublicDataset):
        pub = ds.name
    elif isinstance(ds, ComposedDataset):
        fp = str(ds.source.path)
        cdt = str(ds.source.format) if ds.source.format else None
        samp = ds.source.sampling
    if prof and prof.type == PhaseType.FIXED_SCHEDULE:
        fixed, auto_off = True, prof.auto_offset
        start_off, end_off = prof.start_offset, prof.end_offset
    synth = SynthesisConfig()
    if isinstance(ds, FileDataset) and ds.synthesis is not None:
        s = ds.synthesis
        synth = SynthesisConfig(
            speedup_ratio=s.speedup_ratio,
            prefix_len_multiplier=s.prefix_len_multiplier,
            prefix_root_multiplier=s.prefix_root_multiplier,
            prompt_len_multiplier=s.prompt_len_multiplier,
            max_isl=s.max_isl,
            max_osl=s.max_osl,
        )
    ad = _multimodal(ds, "audio")
    if ad:
        alm, als = _ms(ad.length)
        audio = AudioConfig(
            batch_size=ad.batch_size,
            format=ad.format,
            length=AudioLengthConfig(mean=alm or 0.0, stddev=als or 0.0),
            sample_rates=ad.sample_rates,
            depths=ad.depths,
            num_channels=ad.channels,
        )
    else:
        audio = AudioConfig()
    imd = _multimodal(ds, "images")
    if imd:
        wm, ws = _ms(imd.width)
        hm, hs = _ms(imd.height)
        image = ImageConfig(
            batch_size=imd.batch_size,
            format=imd.format,
            width=ImageWidthConfig(mean=wm or 0.0, stddev=ws or 0.0),
            height=ImageHeightConfig(mean=hm or 0.0, stddev=hs or 0.0),
        )
    else:
        image = ImageConfig()
    vd = _multimodal(ds, "video")
    video = (
        VideoConfig(batch_size=0, duration=1.0)
        if not vd
        else VideoConfig(
            batch_size=vd.batch_size,
            duration=vd.duration,
            fps=vd.fps,
            width=vd.width,
            height=vd.height,
            format=vd.format,
            codec=vd.codec,
            synth_type=vd.synth_type,
        )
    )
    rankings = RankingsConfig()
    if isinstance(ds, SyntheticDataset) and ds.rankings is not None:
        r = ds.rankings
        pm, ps = _ms(r.passages)
        ptm, pts = _ms(r.passage_tokens)
        qtm, qts = _ms(r.query_tokens)
        p_kw = _omit(
            mean=_int(pm),
            stddev=_int(ps),
            prompt_token_mean=_int(ptm),
            prompt_token_stddev=_int(pts),
        )
        q_kw = _omit(prompt_token_mean=_int(qtm), prompt_token_stddev=_int(qts))
        rankings = RankingsConfig(
            passages=RankingsPassagesConfig(**p_kw),
            query=RankingsQueryConfig(**q_kw),
        )
    prefix = PrefixPromptConfig()
    if isinstance(ds, SyntheticDataset) and ds.prefix_prompts is not None:
        pp = ds.prefix_prompts
        prefix = PrefixPromptConfig(
            length=pp.length or 0,
            pool_size=pp.pool_size or 0,
            shared_system_prompt_length=pp.shared_system_length,
            user_context_prompt_length=pp.user_context_length,
        )
    elif isinstance(ds, ComposedDataset) and ds.augment.prefix is not None:
        lm, _ = _ms(ds.augment.prefix.length)
        prefix = PrefixPromptConfig(
            length=int(lm) if lm else 0,
            pool_size=ds.augment.prefix.pool_size or 0,
        )
    if isinstance(ds, SyntheticDataset) and ds.prompts is not None:
        p = ds.prompts
        im, is_ = _ms(p.isl)
        om, os_ = _ms(p.osl)
        prompt = PromptConfig(
            input_tokens=InputTokensConfig(
                **_omit(
                    mean=_int(im),
                    stddev=float(is_) if is_ is not None else None,
                    block_size=p.block_size or None,
                )
            ),
            output_tokens=OutputTokensConfig(**_omit(mean=_int(om), stddev=_int(os_))),
            prefix_prompt=prefix,
            sequence_distribution=p.sequence_distribution,
        )
    elif isinstance(ds, ComposedDataset):
        om, os_ = _ms(ds.augment.osl)
        prompt = PromptConfig(
            output_tokens=OutputTokensConfig(**_omit(mean=_int(om), stddev=_int(os_))),
            prefix_prompt=prefix,
        )
    else:
        prompt = PromptConfig(prefix_prompt=prefix)
    if isinstance(ds, SyntheticDataset):
        ctm, cts = _ms(ds.turns)
        cdm, cds = _ms(ds.turn_delay)
        turn = TurnConfig(
            **_omit(mean=_int(ctm), stddev=_int(cts)),
            delay=TurnDelayConfig(
                **_omit(mean=cdm, stddev=cds), ratio=ds.turn_delay_ratio
            ),
        )
    else:
        turn = TurnConfig()
    conversation = ConversationConfig(
        turn=turn,
        **_omit(num_dataset_entries=ds.entries, num=prof.sessions if prof else None),
    )
    return InputConfig.model_construct(
        extra=list(config.endpoint.extra.items()) if config.endpoint.extra else None,
        headers=list(config.endpoint.headers.items())
        if config.endpoint.headers
        else None,
        file=fp,
        fixed_schedule=fixed,
        fixed_schedule_auto_offset=auto_off,
        fixed_schedule_start_offset=start_off,
        fixed_schedule_end_offset=end_off,
        public_dataset=pub,
        custom_dataset_type=cdt,
        dataset_sampling_strategy=samp,
        random_seed=config.random_seed,
        goodput=dict(config.slos) if config.slos else None,
        audio=audio,
        image=image,
        video=video,
        prompt=prompt,
        rankings=rankings,
        synthesis=synth,
        conversation=conversation,
    )


def _build_accuracy(config: AIPerfConfig) -> LegacyAccuracyConfig:
    acc = config.accuracy
    if acc is None:
        return LegacyAccuracyConfig()
    return LegacyAccuracyConfig(
        benchmark=acc.benchmark,
        tasks=acc.tasks,
        n_shots=acc.n_shots,
        enable_cot=acc.enable_cot,
        grader=acc.grader,
        system_prompt=acc.system_prompt,
        verbose=acc.verbose,
    )


def _build_loadgen(config: AIPerfConfig, prof: PhaseConfig | None):
    kw: dict[str, Any] = {"benchmark_grace_period": float("inf")}
    warmup = next((p for p in config.load.values() if p.exclude), None)
    for phase, w in [(prof, ""), (warmup, "warmup_")]:
        if phase is None:
            continue
        dur_key = f"{w}duration" if w else "benchmark_duration"
        grace_key = f"{w}grace_period" if w else "benchmark_grace_period"
        kw |= {
            f"{w}request_count": phase.requests,
            dur_key: phase.duration,
            f"{w}concurrency": phase.concurrency,
            f"{w}prefill_concurrency": phase.prefill_concurrency,
            grace_key: phase.grace_period
            if (w or phase.grace_period is not None)
            else float("inf"),
            f"{w}concurrency_ramp_duration": _ramp(phase.concurrency_ramp),
            f"{w}prefill_concurrency_ramp_duration": _ramp(phase.prefill_ramp),
            f"{w}request_rate_ramp_duration": _ramp(phase.rate_ramp),
        }
        if w:
            kw[f"{w}num_sessions"] = phase.sessions
        if phase.type == PhaseType.USER_CENTRIC and not w:
            kw |= {"user_centric_rate": phase.rate, "num_users": phase.users}
        elif phase.type == PhaseType.CONCURRENCY:
            kw[f"{w}arrival_pattern"] = ArrivalPattern.CONCURRENCY_BURST
        elif phase.type != PhaseType.USER_CENTRIC:
            kw[f"{w}request_rate"] = phase.rate
            kw[f"{w}arrival_pattern"] = {
                PhaseType.POISSON: ArrivalPattern.POISSON,
                PhaseType.GAMMA: ArrivalPattern.GAMMA,
                PhaseType.CONSTANT: ArrivalPattern.CONSTANT,
            }.get(phase.type, ArrivalPattern.POISSON)
            if not w:
                kw["arrival_smoothness"] = phase.smoothness
        if not w and phase.cancellation:
            kw |= {
                "request_cancellation_rate": phase.cancellation.rate,
                "request_cancellation_delay": phase.cancellation.delay,
            }
    mr = config.multi_run
    kw["num_profile_runs"] = mr.num_runs
    if mr.num_runs > 1:
        kw |= {
            "profile_run_cooldown_seconds": mr.cooldown_seconds,
            "confidence_level": mr.confidence_level,
            "profile_run_disable_warmup_after_first": mr.disable_warmup_after_first,
            "set_consistent_seed": mr.set_consistent_seed,
        }
    return LoadGeneratorConfig(**kw)
