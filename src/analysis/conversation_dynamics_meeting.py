#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _load_transcript(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load transcript_final.json; handle both list and {'segments': [...]}."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read JSON: {p}\n{e}")

    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    raise SystemExit(f"Unexpected transcript format in {p}")


def _compute_meeting_bounds(segments: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Return (start, end) of the meeting."""
    if not segments:
        return 0.0, 0.0
    starts = []
    ends = []
    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        starts.append(s)
        ends.append(e)
    return min(starts), max(ends)


def _normalize_timeline(
    segments: List[Dict[str, Any]],
    meeting_start: float,
    meeting_end: float,
    min_silence: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Rebuild the timeline so that:
      - speech segments are kept as is
      - SILENCE segments are recomputed as the complement of any speech
        across all speakers within [meeting_start, meeting_end]

    This guarantees that silence never overlaps with speech.
    """

    # Keep only speech segments from the original
    speech = [
        dict(seg) for seg in segments
        if seg.get("type", "speech") == "speech"
    ]

    if not speech:
        dur = max(0.0, meeting_end - meeting_start)
        if dur >= min_silence:
            return [{
                "type": "silence",
                "start": float(meeting_start),
                "end": float(meeting_end),
            }]
        return []

    # Sort by start time
    speech.sort(key=lambda s: float(s.get("start", 0.0)))

    # Build merged "anyone is speaking" intervals
    merged: List[Dict[str, float]] = []
    for seg in speech:
        st = max(float(seg.get("start", 0.0)), meeting_start)
        en = min(float(seg.get("end", st)), meeting_end)
        if not merged:
            merged.append({"start": st, "end": en})
        else:
            last = merged[-1]
            if st <= last["end"]:  # overlap or touch
                last["end"] = max(last["end"], en)
            else:
                merged.append({"start": st, "end": en})

    # Compute silence intervals as gaps between merged speech windows
    silences: List[Dict[str, Any]] = []
    prev_end = meeting_start

    for win in merged:
        gap_start = prev_end
        gap_end = win["start"]
        if gap_end - gap_start >= min_silence:
            silences.append({
                "type": "silence",
                "start": round(gap_start, 2),
                "end": round(gap_end, 2),
            })
        prev_end = max(prev_end, win["end"])

    # Trailing silence
    if meeting_end - prev_end >= min_silence:
        silences.append({
            "type": "silence",
            "start": round(prev_end, 2),
            "end": round(float(meeting_end), 2),
        })

    # Combine speech segments (as speech) plus rebuilt silence and sort by time
    cleaned: List[Dict[str, Any]] = []
    for seg in speech:
        seg = dict(seg)
        seg["type"] = "speech"
        cleaned.append(seg)

    cleaned.extend(silences)
    cleaned.sort(key=lambda s: float(s.get("start", 0.0)))
    return cleaned


def _compute_participation(
    segments: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], float, float]:
    """
    Compute per speaker participation and overall speech/silence time.

    Returns:
      speakers_stats: {speaker: {...}}
      total_speech_time_sec
      total_silence_time_sec
    """
    speakers_stats: Dict[str, Dict[str, Any]] = {}
    total_speech_time = 0.0
    total_silence_time = 0.0

    for seg in segments:
        seg_type = seg.get("type", "speech")
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        dur = max(0.0, e - s)

        if seg_type == "speech":
            spk = str(seg.get("speaker", "S?"))
            total_speech_time += dur
            if spk not in speakers_stats:
                speakers_stats[spk] = {
                    "total_speaking_time_sec": 0.0,
                    "num_turns": 0,
                    "avg_turn_length_sec": 0.0,  # filled after
                    "longest_streak_time_sec": 0.0,  # filled later
                }
            speakers_stats[spk]["total_speaking_time_sec"] += dur
            speakers_stats[spk]["num_turns"] += 1
        elif seg_type == "silence":
            total_silence_time += dur
        else:
            # unknown type; ignore for now
            continue

    return speakers_stats, total_speech_time, total_silence_time


def _compute_turn_changes(
    segments: List[Dict[str, Any]]
) -> int:
    """
    Compute total_turns = total number of speaker changes.

    Only considers speech segments in time order.
    """
    speech_segs = [
        seg for seg in segments if seg.get("type", "speech") == "speech"
    ]
    speech_segs.sort(key=lambda s: float(s.get("start", 0.0)))

    total_turns = 0
    prev_speaker: Optional[str] = None

    for seg in speech_segs:
        spk = str(seg.get("speaker", "S?"))
        if prev_speaker is None:
            prev_speaker = spk
            continue
        if spk != prev_speaker:
            total_turns += 1
            prev_speaker = spk

    return total_turns


def _compute_longest_streaks(
    segments: List[Dict[str, Any]],
    speakers_stats: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute longest uninterrupted speaking streak per speaker and global max.

    Silence or a different speaker breaks the streak.
    Returns:
      {
        "per_speaker": { spk: {"duration_sec": ..., "start": ..., "end": ...}, ... },
        "global": {"speaker": ..., "duration_sec": ..., "start": ..., "end": ...}
      }
    """
    segs_sorted = sorted(segments, key=lambda s: float(s.get("start", 0.0)))

    per_spk: Dict[str, Dict[str, Any]] = {
        spk: {"duration_sec": 0.0, "start": None, "end": None}
        for spk in speakers_stats.keys()
    }

    global_best = {"speaker": None, "duration_sec": 0.0, "start": None, "end": None}

    current_spk: Optional[str] = None
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    def close_streak():
        nonlocal current_spk, current_start, current_end, per_spk, global_best
        if current_spk is None or current_start is None or current_end is None:
            return
        dur = max(0.0, current_end - current_start)
        # update per speaker
        sp_info = per_spk.get(current_spk)
        if sp_info is None:
            sp_info = {"duration_sec": 0.0, "start": None, "end": None}
            per_spk[current_spk] = sp_info
        if dur > sp_info["duration_sec"]:
            sp_info["duration_sec"] = dur
            sp_info["start"] = current_start
            sp_info["end"] = current_end
        # update global
        if dur > global_best["duration_sec"]:
            global_best["duration_sec"] = dur
            global_best["speaker"] = current_spk
            global_best["start"] = current_start
            global_best["end"] = current_end
        # reset
        current_spk = None
        current_start = None
        current_end = None

    for seg in segs_sorted:
        seg_type = seg.get("type", "speech")
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))

        if seg_type != "speech":
            # silence or other types break any ongoing streak
            close_streak()
            continue

        spk = str(seg.get("speaker", "S?"))

        if current_spk is None:
            # start new streak
            current_spk = spk
            current_start = s
            current_end = e
        elif spk == current_spk:
            # extend streak
            current_end = max(current_end, e)
        else:
            # speaker change breaks streak
            close_streak()
            # start new streak for new speaker
            current_spk = spk
            current_start = s
            current_end = e

    # close last streak if any
    close_streak()

    # also fill longest_streak_time_sec into speakers_stats for consistency
    for spk, info in per_spk.items():
        speakers_stats[spk]["longest_streak_time_sec"] = float(
            info.get("duration_sec") or 0.0
        )

    return {"per_speaker": per_spk, "global": global_best}


def _finalize_speaker_percentages_and_avg(
    speakers_stats: Dict[str, Dict[str, Any]], total_meeting_duration: float
) -> None:
    """Add speaking_time_pct_of_meeting and avg_turn_length_sec for each speaker."""
    for spk, stats in speakers_stats.items():
        total_t = float(stats.get("total_speaking_time_sec", 0.0))
        num_turns = int(stats.get("num_turns", 0))
        if total_meeting_duration > 0.0:
            stats["speaking_time_pct_of_meeting"] = total_t / total_meeting_duration
        else:
            stats["speaking_time_pct_of_meeting"] = 0.0
        stats["avg_turn_length_sec"] = (
            total_t / num_turns if num_turns > 0 else 0.0
        )


def _speaker_order_for_plot(segments: List[Dict[str, Any]]) -> List[str]:
    """
    Decide y axis order for speakers in plot: first actual speakers, then 'SILENCE' if present.
    """
    seen = set()
    order: List[str] = []
    for seg in sorted(segments, key=lambda x: (float(x.get("start", 0.0)),
                                               float(x.get("end", 0.0)))):
        seg_type = seg.get("type", "speech")
        if seg_type == "speech":
            spk = str(seg.get("speaker", "S?"))
        elif seg_type == "silence":
            spk = "SILENCE"
        else:
            continue
        if spk not in seen:
            seen.add(spk)
            order.append(spk)
    return order


def _plot_timeline(
    segments: List[Dict[str, Any]],
    meeting_id: str,
    title: str,
    save_path: Union[str, Path],
    min_visible: float = 0.08,
) -> None:
    """
    Plot a timeline per speaker plus SILENCE row using transcript segments.
    """
    if not segments:
        print("[warn] no segments to plot.")
        return

    # Create pseudo speaker "SILENCE" rows from silence segments
    plot_segs: List[Dict[str, Any]] = []
    for seg in segments:
        seg_type = seg.get("type", "speech")
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))

        if seg_type == "speech":
            plot_segs.append(
                {
                    "start": s,
                    "end": e,
                    "speaker": str(seg.get("speaker", "S?")),
                    "type": "speech",
                }
            )
        elif seg_type == "silence":
            plot_segs.append(
                {
                    "start": s,
                    "end": e,
                    "speaker": "SILENCE",
                    "type": "silence",
                }
            )

    if not plot_segs:
        print("[warn] no speech or silence segments to plot.")
        return

    order = _speaker_order_for_plot(segments)
    y_index = {spk: i for i, spk in enumerate(order)}

    cmap = plt.get_cmap("tab20")
    colors = {spk: cmap(i % 20) for i, spk in enumerate(order)}

    fig_h = max(3.0, 0.7 * len(order))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    max_time = 0.0
    seg_count = 0

    for seg in plot_segs:
        spk = seg.get("speaker", "S?")
        y = y_index.get(spk, 0)
        x0 = float(seg.get("start", 0.0))
        x1 = float(seg.get("end", x0))
        w = max(min_visible, x1 - x0)

        max_time = max(max_time, x1)
        seg_count += 1

        rect = patches.Rectangle(
            (x0, y - 0.35),
            w,
            0.7,
            linewidth=0,
            facecolor=colors[spk],
            alpha=0.75,
        )
        ax.add_patch(rect)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Time (s)")
    ax.set_title(title or f"Conversation Timeline: {meeting_id}")
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(left=0, right=max(max_time, 1.0))
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    for i in range(len(order)):
        ax.hlines(i, 0, max_time, linewidth=0.5, alpha=0.2)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved timeline plot → {save_path}")
    print(f"[summary] speakers (including SILENCE if present): {len(order)} → {order}")
    print(f"[summary] segments plotted: {seg_count}, duration ~ {max_time:.1f}s")


def _write_json_summary(
    path: Union[str, Path],
    meeting_id: str,
    total_meeting_duration: float,
    total_speech_time: float,
    total_silence_time: float,
    speakers_stats: Dict[str, Dict[str, Any]],
    total_turns: int,
    streaks_info: Dict[str, Any],
) -> None:
    payload = {
        "metadata": {
            "meeting_id": meeting_id,
            "total_meeting_duration_sec": total_meeting_duration,
            "total_speech_time_sec": total_speech_time,
            "total_silence_time_sec": total_silence_time,
            "num_speakers": len(speakers_stats),
            "context": "conversation_dynamics_meeting_v1",
        },
        "speakers": speakers_stats,
        "turn_taking": {
            "total_turns": total_turns,
            "longest_streak": streaks_info.get("global", {}),
        },
        "streaks_per_speaker": streaks_info.get("per_speaker", {}),
    }
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote JSON summary → {path}")


def _write_text_summary(
    path: Union[str, Path],
    meeting_id: str,
    total_meeting_duration: float,
    total_speech_time: float,
    total_silence_time: float,
    speakers_stats: Dict[str, Dict[str, Any]],
    total_turns: int,
    streaks_info: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append(f"Meeting ID: {meeting_id}")
    lines.append(f"Total duration: {total_meeting_duration:.2f} s")
    lines.append(f"Total speech time: {total_speech_time:.2f} s")
    lines.append(f"Total silence time: {total_silence_time:.2f} s")
    lines.append("")
    lines.append("Speakers:")

    for spk, stats in sorted(speakers_stats.items()):
        t = float(stats.get("total_speaking_time_sec", 0.0))
        pct = float(stats.get("speaking_time_pct_of_meeting", 0.0)) * 100.0
        nturns = int(stats.get("num_turns", 0))
        avg_turn = float(stats.get("avg_turn_length_sec", 0.0))
        longest = float(stats.get("longest_streak_time_sec", 0.0))
        lines.append(
            f"- {spk}: {t:.2f} s ({pct:.1f}%), {nturns} turns, "
            f"avg turn {avg_turn:.2f} s, longest streak {longest:.2f} s"
        )

    lines.append("")
    lines.append(f"Total speaker changes (total_turns): {total_turns}")

    global_streak = streaks_info.get("global", {})
    gs_spk = global_streak.get("speaker")
    gs_dur = global_streak.get("duration_sec")
    gs_start = global_streak.get("start")
    gs_end = global_streak.get("end")

    if gs_spk is not None and gs_dur is not None:
        lines.append("")
        lines.append("Longest uninterrupted streak overall:")
        lines.append(
            f"- Speaker: {gs_spk}, {gs_dur:.2f} s "
            f"(from {gs_start:.2f} to {gs_end:.2f})"
        )

    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote text summary → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute meeting level conversation dynamics and timeline plot."
    )
    ap.add_argument(
        "--meeting-id",
        required=True,
        help="Meeting ID under data/outputs/<MEETING_ID>/",
    )
    ap.add_argument(
        "--title",
        default="Conversation Timeline",
        help="Plot title.",
    )
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    transcript_path = base / "transcript.json"
    if not transcript_path.exists():
        raise SystemExit(f"Missing transcript file: {transcript_path}")

    segments = _load_transcript(transcript_path)
    if not segments:
        raise SystemExit(f"No segments found in {transcript_path}")

    # Meeting bounds from raw transcript
    meeting_start, meeting_end = _compute_meeting_bounds(segments)
    total_meeting_duration = max(0.0, meeting_end - meeting_start)

    # Rebuild silence so it never overlaps with speech
    segments = _normalize_timeline(
        segments,
        meeting_start=meeting_start,
        meeting_end=meeting_end,
        min_silence=0.2,
    )

    # Participation per speaker
    speakers_stats, total_speech_time, total_silence_time = _compute_participation(
        segments
    )

    # Longest streaks (also fills longest_streak_time_sec in speakers_stats)
    streaks_info = _compute_longest_streaks(segments, speakers_stats)

    # Percentages and average turn length
    _finalize_speaker_percentages_and_avg(speakers_stats, total_meeting_duration)

    # Total speaker changes
    total_turns = _compute_turn_changes(segments)

    # JSON and text summaries
    json_out = base / "conversation_dynamics_meeting.json"
    txt_out = base / "conversation_dynamics_meeting.txt"
    _write_json_summary(
        json_out,
        args.meeting_id,
        total_meeting_duration,
        total_speech_time,
        total_silence_time,
        speakers_stats,
        total_turns,
        streaks_info,
    )
    _write_text_summary(
        txt_out,
        args.meeting_id,
        total_meeting_duration,
        total_speech_time,
        total_silence_time,
        speakers_stats,
        total_turns,
        streaks_info,
    )

    # Timeline plot with rebuilt silence
    plots_dir = base / "plots"
    out_plot = plots_dir / f"{args.meeting_id}_timeline.png"
    _plot_timeline(segments, args.meeting_id, args.title, out_plot)


if __name__ == "__main__":
    main()
