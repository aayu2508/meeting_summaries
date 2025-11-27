#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import math
from collections import defaultdict

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Loading and basic helpers
# ------------------------------------------------------------

def _load_segments(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load merged transcript JSON.
    Handles:
      - list of segments
      - {"segments": [...]} wrapper
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read JSON: {p}\n{e}")

    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    raise SystemExit(f"Unexpected JSON format in {p}")


def _compute_meeting_bounds(segments: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Return (start, end) of the meeting from any segments."""
    if not segments:
        return 0.0, 0.0
    starts, ends = [], []
    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        starts.append(s)
        ends.append(e)
    return min(starts), max(ends)


def _safe_get_emotion(seg: Dict[str, Any]) -> Optional[Dict[str, float]]:
    emo = seg.get("emotion") or {}
    try:
        v = emo.get("v", None)
        a01 = emo.get("a01", None)
        d01 = emo.get("d01", None)
        if v is None or a01 is None or d01 is None:
            return None
        return {"v": float(v), "a01": float(a01), "d01": float(d01)}
    except Exception:
        return None


def _safe_get_log_rms(seg: Dict[str, Any]) -> Optional[float]:
    prosody = seg.get("prosody") or {}
    val = prosody.get("log_rms", None)
    if val is None:
        return None
    try:
        x = float(val)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


# ------------------------------------------------------------
# 1. Emotional jumps
# ------------------------------------------------------------

def _find_emotional_jumps(
    segments: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Find top K emotional jumps between consecutive speech segments.

    Uses absolute differences in:
      - v (valence)
      - a01 (arousal)
      - d01 (dominance)
    """
    speech_idxs = [i for i, s in enumerate(segments) if s.get("type") == "speech"]

    jumps: List[Dict[str, Any]] = []
    for i_prev, i_curr in zip(speech_idxs, speech_idxs[1:]):
        s_prev = segments[i_prev]
        s_curr = segments[i_curr]

        emo_prev = _safe_get_emotion(s_prev)
        emo_curr = _safe_get_emotion(s_curr)
        if emo_prev is None or emo_curr is None:
            continue

        dv = abs(emo_curr["v"]   - emo_prev["v"])
        da = abs(emo_curr["a01"] - emo_prev["a01"])
        dd = abs(emo_curr["d01"] - emo_prev["d01"])
        magnitude = math.sqrt(dv * dv + da * da + dd * dd)

        jumps.append({
            "from_index": i_prev,
            "to_index": i_curr,
            "from_segment_id": s_prev.get("segment_id"),
            "to_segment_id": s_curr.get("segment_id"),
            "from_speaker": s_prev.get("speaker"),
            "to_speaker": s_curr.get("speaker"),
            "from_start": float(s_prev.get("start", 0.0)),
            "to_start": float(s_curr.get("start", 0.0)),
            "dv": dv,
            "da": da,
            "dd": dd,
            "magnitude": magnitude,
        })

    jumps.sort(key=lambda x: x["magnitude"], reverse=True)
    return jumps[:top_k]


# ------------------------------------------------------------
# 2. Speaker loudness (log_rms)
# ------------------------------------------------------------

def _compute_speaker_loudness(
    segments: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute loudness stats per speaker from prosody.log_rms.

    Returns dict[speaker] = {
      "count_segments",
      "total_duration",
      "mean_log_rms_unweighted",
      "mean_log_rms_weighted",
      "std_log_rms",
      "pct_segments_above_global_mean",
      "global_mean_log_rms"
    }
    """
    per_speaker: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    all_values: List[float] = []

    for seg in segments:
        if seg.get("type") != "speech":
            continue
        speaker = seg.get("speaker")
        if not speaker:
            continue

        log_rms = _safe_get_log_rms(seg)
        if log_rms is None:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        dur = max(0.0, end - start)

        per_speaker[speaker].append((log_rms, dur))
        all_values.append(log_rms)

    if not all_values:
        return {}

    global_mean = sum(all_values) / len(all_values)

    stats: Dict[str, Dict[str, float]] = {}
    for speaker, vals in per_speaker.items():
        logs = [v for v, _ in vals]
        durs = [d for _, d in vals]

        n = len(logs)
        total_dur = sum(durs) if durs else 0.0
        mean_unweighted = sum(logs) / n if n > 0 else 0.0

        if total_dur > 0:
            mean_weighted = sum(l * d for (l, d) in vals) / total_dur
        else:
            mean_weighted = mean_unweighted

        if n > 1:
            mu = mean_unweighted
            var = sum((x - mu) ** 2 for x in logs) / (n - 1)
            std = math.sqrt(max(var, 0.0))
        else:
            std = 0.0

        above_global = sum(1 for x in logs if x > global_mean)
        pct_above_global = 100.0 * above_global / n if n > 0 else 0.0

        stats[speaker] = {
            "count_segments": float(n),
            "total_duration": float(total_dur),
            "mean_log_rms_unweighted": float(mean_unweighted),
            "mean_log_rms_weighted": float(mean_weighted),
            "std_log_rms": float(std),
            "pct_segments_above_global_mean": float(pct_above_global),
            "global_mean_log_rms": float(global_mean),
        }

    return stats


# ------------------------------------------------------------
# 3. Silence and pause analysis
# ------------------------------------------------------------

def _analyze_meeting_silences(
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Meeting level silence summary from explicit silence segments.
    """
    silence_segments = [s for s in segments if s.get("type") == "silence"]

    total_silence = 0.0
    longest = 0.0
    longest_start = None
    all_silences: List[Dict[str, float]] = []

    for s in silence_segments:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        dur = max(0.0, end - start)

        total_silence += dur
        all_silences.append({"start": start, "end": end, "duration": dur})
        if dur > longest:
            longest = dur
            longest_start = start

    if segments:
        meeting_end = max(float(s.get("end", 0.0)) for s in segments)
        meeting_start = min(float(s.get("start", 0.0)) for s in segments)
    else:
        meeting_end = 0.0
        meeting_start = 0.0

    meeting_duration = max(0.0, meeting_end - meeting_start)
    silence_fraction = total_silence / meeting_duration if meeting_duration > 0 else 0.0

    summary = {
        "total_silence_time": float(total_silence),
        "silence_fraction": float(silence_fraction),
        "num_silences": float(len(all_silences)),
        "longest_silence": float(longest),
        "longest_silence_start": float(longest_start) if longest_start is not None else None,
        "meeting_duration": float(meeting_duration),
    }

    return {
        "summary": summary,
        "all_silences": all_silences,
    }


def _analyze_speaker_pauses(
    segments: List[Dict[str, Any]],
    long_threshold: float = 1.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Analyze pauses before and after speakers talk using segment gaps.

    For each speech segment:
      - pause_before: gap between previous segment end and this start (if > 0)
      - pause_after:  gap between this end and next segment start (if > 0)

    Returns:
      speaker_pauses: { speaker: {...stats...} }
      long_pause_events: list of pause events >= long_threshold
    """
    segs_sorted = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    n = len(segs_sorted)

    per_speaker_before: Dict[str, List[float]] = defaultdict(list)
    per_speaker_after: Dict[str, List[float]] = defaultdict(list)
    per_speaker_before_long: Dict[str, int] = defaultdict(int)
    per_speaker_after_long: Dict[str, int] = defaultdict(int)
    long_events: List[Dict[str, Any]] = []

    for i, seg in enumerate(segs_sorted):
        if seg.get("type") != "speech":
            continue
        speaker = str(seg.get("speaker", "S?"))
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))

        # pause before
        if i > 0:
            prev_end = float(segs_sorted[i - 1].get("end", start))
            gap_before = max(0.0, start - prev_end)
            if gap_before > 0.0:
                per_speaker_before[speaker].append(gap_before)
                if gap_before >= long_threshold:
                    per_speaker_before_long[speaker] += 1
                    long_events.append({
                        "type": "before",
                        "speaker": speaker,
                        "start": prev_end,
                        "end": start,
                        "duration": gap_before,
                    })

        # pause after
        if i < n - 1:
            next_start = float(segs_sorted[i + 1].get("start", end))
            gap_after = max(0.0, next_start - end)
            if gap_after > 0.0:
                per_speaker_after[speaker].append(gap_after)
                if gap_after >= long_threshold:
                    per_speaker_after_long[speaker] += 1
                    long_events.append({
                        "type": "after",
                        "speaker": speaker,
                        "start": end,
                        "end": next_start,
                        "duration": gap_after,
                    })

    speaker_pauses: Dict[str, Any] = {}
    all_speakers = set(list(per_speaker_before.keys()) + list(per_speaker_after.keys()))
    for speaker in sorted(all_speakers):
        before_list = per_speaker_before.get(speaker, [])
        after_list = per_speaker_after.get(speaker, [])

        def _mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        speaker_pauses[speaker] = {
            "avg_pause_before": float(_mean(before_list)),
            "max_pause_before": float(max(before_list) if before_list else 0.0),
            "count_pause_before": int(len(before_list)),
            "count_pause_before_long": int(per_speaker_before_long.get(speaker, 0)),
            "avg_pause_after": float(_mean(after_list)),
            "max_pause_after": float(max(after_list) if after_list else 0.0),
            "count_pause_after": int(len(after_list)),
            "count_pause_after_long": int(per_speaker_after_long.get(speaker, 0)),
            "long_threshold_sec": float(long_threshold),
        }

    return speaker_pauses, long_events


# ------------------------------------------------------------
# Formatting for text report
# ------------------------------------------------------------

def _format_emotional_jumps(jumps: List[Dict[str, Any]], max_print: int = 10) -> str:
    lines: List[str] = []
    if not jumps:
        lines.append("No emotional jumps found.")
        return "\n".join(lines)

    lines.append("Top emotional jumps:")
    for j in jumps[:max_print]:
        t1 = j["from_start"]
        t2 = j["to_start"]
        lines.append(
            f"  {t1:7.2f}s -> {t2:7.2f}s | "
            f"{j['from_speaker']} -> {j['to_speaker']} | "
            f"dv={j['dv']:.3f}, da={j['da']:.3f}, dd={j['dd']:.3f}, "
            f"mag={j['magnitude']:.3f}"
        )
    return "\n".join(lines)


def _format_speaker_loudness(stats: Dict[str, Dict[str, float]]) -> str:
    lines: List[str] = []
    if not stats:
        lines.append("No loudness data available.")
        return "\n".join(lines)

    lines.append("Speaker loudness (prosody.log_rms):")
    speakers_sorted = sorted(
        stats.items(),
        key=lambda kv: kv[1]["mean_log_rms_weighted"],
        reverse=True,
    )
    for speaker, s in speakers_sorted:
        lines.append(
            f"  {speaker:12s} | segs={int(s['count_segments']):3d} | "
            f"dur={s['total_duration']:6.1f}s | "
            f"mean_w={s['mean_log_rms_weighted']:.3f} | "
            f"std={s['std_log_rms']:.3f} | "
            f">%gt_global={s['pct_segments_above_global_mean']:5.1f}%"
        )
    return "\n".join(lines)


def _format_silence_meeting(info: Dict[str, Any]) -> str:
    lines: List[str] = []
    summary = info.get("summary", {})

    lines.append("Meeting silence summary (explicit silence segments):")
    lines.append(
        f"  total_silence_time: {summary.get('total_silence_time', 0.0):6.2f}s"
    )
    lines.append(
        f"  meeting_duration:   {summary.get('meeting_duration', 0.0):6.2f}s"
    )
    lines.append(
        f"  silence_fraction:   "
        f"{summary.get('silence_fraction', 0.0)*100:6.2f}%"
    )
    lines.append(
        f"  num_silences:       {int(summary.get('num_silences', 0))}"
    )
    lines.append(
        f"  longest_silence:    {summary.get('longest_silence', 0.0):6.2f}s"
    )
    ls_start = summary.get("longest_silence_start", None)
    if ls_start is not None:
        lines.append(
            f"  longest_silence_at: {ls_start:6.2f}s"
        )

    return "\n".join(lines)


def _format_speaker_pauses_text(speaker_pauses: Dict[str, Any]) -> str:
    lines: List[str] = []
    if not speaker_pauses:
        lines.append("No speaker pause information available.")
        return "\n".join(lines)

    lines.append("Speaker level pauses (gap before and after speaking):")
    for speaker, stats in sorted(speaker_pauses.items()):
        lines.append(
            f"  {speaker:12s} | "
            f"avg_before={stats['avg_pause_before']:.2f}s "
            f"(max={stats['max_pause_before']:.2f}s, "
            f"n={stats['count_pause_before']}, "
            f"n_long={stats['count_pause_before_long']}) | "
            f"avg_after={stats['avg_pause_after']:.2f}s "
            f"(max={stats['max_pause_after']:.2f}s, "
            f"n={stats['count_pause_after']}, "
            f"n_long={stats['count_pause_after_long']})"
        )

    # Quick highlight: who has longest pause_before / pause_after
    longest_before = max(
        speaker_pauses.items(),
        key=lambda kv: kv[1]["max_pause_before"],
        default=(None, {"max_pause_before": 0.0}),
    )
    longest_after = max(
        speaker_pauses.items(),
        key=lambda kv: kv[1]["max_pause_after"],
        default=(None, {"max_pause_after": 0.0}),
    )

    lines.append("")
    lines.append("Notable silence patterns:")
    if longest_before[0] is not None:
        lines.append(
            f"  Longest pause before speaking: "
            f"{longest_before[0]} (~{longest_before[1]['max_pause_before']:.2f}s)"
        )
    if longest_after[0] is not None:
        lines.append(
            f"  Longest pause after speaking: "
            f"{longest_after[0]} (~{longest_after[1]['max_pause_after']:.2f}s)"
        )

    return "\n".join(lines)


def _write_text_report(
    path: Union[str, Path],
    meeting_id: str,
    meeting_start: float,
    meeting_end: float,
    jumps_text: str,
    loudness_text: str,
    silence_meeting_text: str,
    speaker_pause_text: str,
) -> None:
    lines: List[str] = []
    total_dur = max(0.0, meeting_end - meeting_start)

    lines.append(f"Meeting ID: {meeting_id}")
    lines.append(f"Start time: {meeting_start:.2f} s")
    lines.append(f"End time:   {meeting_end:.2f} s")
    lines.append(f"Duration:   {total_dur:.2f} s")
    lines.append("")
    lines.append("=== Emotional jumps ===")
    lines.append(jumps_text)
    lines.append("")
    lines.append("=== Speaker loudness ===")
    lines.append(loudness_text)
    lines.append("")
    lines.append("=== Meeting silence ===")
    lines.append(silence_meeting_text)
    lines.append("")
    lines.append("=== Speaker pause patterns ===")
    lines.append(speaker_pause_text)

    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote emotion text summary -> {path}")


def _write_json_summary(
    path: Union[str, Path],
    meeting_id: str,
    meeting_start: float,
    meeting_end: float,
    jumps: List[Dict[str, Any]],
    loudness_stats: Dict[str, Dict[str, float]],
    silence_meeting: Dict[str, Any],
    speaker_pauses: Dict[str, Any],
    long_pause_events: List[Dict[str, Any]],
) -> None:
    payload = {
        "metadata": {
            "meeting_id": meeting_id,
            "meeting_start_sec": float(meeting_start),
            "meeting_end_sec": float(meeting_end),
            "total_duration_sec": float(max(0.0, meeting_end - meeting_start)),
            "context": "emotion_dynamics_meeting_v2",
        },
        "emotional_jumps": jumps,
        "speaker_loudness": loudness_stats,
        "meeting_silence": silence_meeting,
        "speaker_pauses": speaker_pauses,
        "long_pause_events": long_pause_events,
    }
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote emotion JSON summary -> {path}")


# ------------------------------------------------------------
# Plotting: updated design
# ------------------------------------------------------------

def _extract_valence_time(
    segments: List[Dict[str, Any]]
) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    vals: List[float] = []
    for seg in segments:
        if seg.get("type") != "speech":
            continue
        emo = _safe_get_emotion(seg)
        if emo is None:
            continue
        t = float(seg.get("start", 0.0))
        times.append(t)
        vals.append(emo["v"])
    return times, vals


def _save_plots(
    segments: List[Dict[str, Any]],
    jumps: List[Dict[str, Any]],
    loudness_stats: Dict[str, Dict[str, float]],
    plots_dir: Path,
    meeting_id: str,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: valence over time + jump magnitudes
    times_v, vals_v = _extract_valence_time(segments)
    if times_v:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True
        )

        # Valence time series
        ax1.plot(times_v, vals_v, marker="o", linestyle="-")
        ax1.set_ylabel("Valence")
        ax1.set_title(f"Valence and emotional jumps over time: {meeting_id}")
        ax1.grid(True, axis="y", alpha=0.3)

        # Mark jump positions (at "to_start")
        if jumps:
            jump_times = [j["to_start"] for j in jumps]
            jump_mags = [j["magnitude"] for j in jumps]
            # Just mark vertical lines at jump times
            for jt in jump_times:
                ax1.axvline(jt, alpha=0.3)

            # Jump magnitudes
            ax2.scatter(jump_times, jump_mags)
            ax2.set_ylabel("Jump magnitude")
            ax2.grid(True, axis="y", alpha=0.3)

        ax2.set_xlabel("Time (s)")
        fig.tight_layout()
        out_path = plots_dir / f"{meeting_id}_emotion_jumps.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] saved plot -> {out_path}")

    # Figure 2: speaker loudness boxplot (log_rms per speaker)
    if loudness_stats:
        # Collect raw log_rms values per speaker from segments
        speaker_vals: Dict[str, List[float]] = defaultdict(list)
        for seg in segments:
            if seg.get("type") != "speech":
                continue
            speaker = seg.get("speaker")
            if not speaker:
                continue
            log_rms = _safe_get_log_rms(seg)
            if log_rms is None:
                continue
            speaker_vals[speaker].append(log_rms)

        if speaker_vals:
            speakers_sorted = sorted(
                speaker_vals.keys(),
                key=lambda s: loudness_stats.get(s, {}).get(
                    "mean_log_rms_weighted", 0.0
                ),
                reverse=True,
            )
            data = [speaker_vals[s] for s in speakers_sorted]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.boxplot(data, showfliers=True)
            ax.set_xticks(range(1, len(speakers_sorted) + 1))
            ax.set_xticklabels(speakers_sorted, rotation=45, ha="right")
            ax.set_ylabel("log_rms")
            ax.set_title(f"Speaker loudness distribution: {meeting_id}")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            out_path = plots_dir / f"{meeting_id}_speaker_loudness_boxplot.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[ok] saved plot -> {out_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Emotion and prosody dynamics for a meeting."
    )
    ap.add_argument(
        "--meeting-id",
        required=True,
        help="Meeting ID under data/outputs/<MEETING_ID>/",
    )
    ap.add_argument(
        "--emotion-json-name",
        default="transcript_emotion.json",
        help="Filename inside data/outputs/<MEETING_ID>/ with merged emotion+prosody JSON.",
    )
    ap.add_argument(
        "--top-jumps",
        type=int,
        default=10,
        help="Number of top emotional jumps to report (default 10).",
    )
    ap.add_argument(
        "--pause-threshold",
        type=float,
        default=1.0,
        help="Duration in seconds to treat a pause as long for per speaker stats.",
    )
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    json_path = base / args.emotion_json_name

    if not json_path.exists():
        raise SystemExit(f"Missing emotion JSON file: {json_path}")

    segments = _load_segments(json_path)
    if not segments:
        raise SystemExit(f"No segments found in {json_path}")

    meeting_start, meeting_end = _compute_meeting_bounds(segments)

    # Core metrics
    jumps = _find_emotional_jumps(segments, top_k=args.top_jumps)
    loudness_stats = _compute_speaker_loudness(segments)
    silence_meeting = _analyze_meeting_silences(segments)
    speaker_pauses, long_pause_events = _analyze_speaker_pauses(
        segments, long_threshold=args.pause_threshold
    )

    # Text summary
    jumps_text = _format_emotional_jumps(jumps, max_print=args.top_jumps)
    loudness_text = _format_speaker_loudness(loudness_stats)
    silence_meeting_text = _format_silence_meeting(silence_meeting)
    speaker_pause_text = _format_speaker_pauses_text(speaker_pauses)

    txt_out = base / "emotion_dynamics_meeting.txt"
    _write_text_report(
        txt_out,
        args.meeting_id,
        meeting_start,
        meeting_end,
        jumps_text,
        loudness_text,
        silence_meeting_text,
        speaker_pause_text,
    )

    # JSON summary
    json_out = base / "emotion_dynamics_meeting.json"
    _write_json_summary(
        json_out,
        args.meeting_id,
        meeting_start,
        meeting_end,
        jumps,
        loudness_stats,
        silence_meeting,
        speaker_pauses,
        long_pause_events,
    )

    # Plots (no silence plots)
    plots_dir = base / "plots"
    _save_plots(
        segments,
        jumps,
        loudness_stats,
        plots_dir,
        args.meeting_id,
    )


if __name__ == "__main__":
    main()
