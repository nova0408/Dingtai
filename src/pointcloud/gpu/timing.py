# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class GpuTimingEvent:
    name: str
    upload_ms: float
    execute_ms: float
    download_ms: float
    total_ms: float
    cache_hit: bool | None = None
    backend: str | None = None
    tag: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class GpuTimingAggregate:
    count: int
    upload_ms_sum: float
    execute_ms_sum: float
    download_ms_sum: float
    total_ms_sum: float
    upload_ms_avg: float
    execute_ms_avg: float
    download_ms_avg: float
    total_ms_avg: float

    @classmethod
    def from_events(cls, events: list[GpuTimingEvent]) -> GpuTimingAggregate:
        count = len(events)
        upload_sum = sum(x.upload_ms for x in events)
        execute_sum = sum(x.execute_ms for x in events)
        download_sum = sum(x.download_ms for x in events)
        total_sum = sum(x.total_ms for x in events)
        if count <= 0:
            return cls(
                count=0,
                upload_ms_sum=0.0,
                execute_ms_sum=0.0,
                download_ms_sum=0.0,
                total_ms_sum=0.0,
                upload_ms_avg=0.0,
                execute_ms_avg=0.0,
                download_ms_avg=0.0,
                total_ms_avg=0.0,
            )
        return cls(
            count=count,
            upload_ms_sum=upload_sum,
            execute_ms_sum=execute_sum,
            download_ms_sum=download_sum,
            total_ms_sum=total_sum,
            upload_ms_avg=upload_sum / count,
            execute_ms_avg=execute_sum / count,
            download_ms_avg=download_sum / count,
            total_ms_avg=total_sum / count,
        )


_GPU_TIMING_EVENTS: list[GpuTimingEvent] = []


def reset_gpu_timing_stats() -> None:
    _GPU_TIMING_EVENTS.clear()


def record_gpu_timing_event(
    *,
    name: str,
    upload_ms: float,
    execute_ms: float,
    download_ms: float,
    total_ms: float | None = None,
    cache_hit: bool | None = None,
    backend: str | None = None,
    tag: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    resolved_total_ms = total_ms
    if resolved_total_ms is None:
        resolved_total_ms = upload_ms + execute_ms + download_ms
    event = GpuTimingEvent(
        name=name,
        upload_ms=upload_ms,
        execute_ms=execute_ms,
        download_ms=download_ms,
        total_ms=resolved_total_ms,
        cache_hit=cache_hit,
        backend=backend,
        tag=tag,
        extra=dict(extra) if extra is not None else {},
    )
    _GPU_TIMING_EVENTS.append(event)


def snapshot_gpu_timing_stats() -> dict[str, object]:
    events = list(_GPU_TIMING_EVENTS)
    overall = GpuTimingAggregate.from_events(events)
    by_name_groups: dict[str, list[GpuTimingEvent]] = {}
    for event in events:
        group = by_name_groups.setdefault(event.name, [])
        group.append(event)
    by_name_summary: dict[str, dict[str, object]] = {
        name: asdict(GpuTimingAggregate.from_events(group))
        for name, group in by_name_groups.items()
    }
    event_payload: list[dict[str, object]] = []
    for event in events:
        item = asdict(event)
        extra = item.pop("extra", {})
        item.update(extra)
        event_payload.append(item)
    return {
        "overall": asdict(overall),
        "by_name": by_name_summary,
        "events": event_payload,
    }
