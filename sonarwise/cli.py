"""sonarwise CLI — command line interface.

Usage:
    sonarwise index meeting.wav
    sonarwise query "budget discussion"
    sonarwise speakers meeting.wav
    sonarwise listen --source microphone
"""

from __future__ import annotations

import argparse
import json
import sys

from sonarwise.utils.time_utils import ms_to_short


def main():
    parser = argparse.ArgumentParser(
        prog="sonarwise",
        description="sonarwise - Pluggable audio perception engine. Hear. Search. Retrieve.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ─── index ───
    idx = subparsers.add_parser("index", help="Index audio file(s)")
    idx.add_argument("path", help="Audio file or folder path")
    idx.add_argument("--recursive", "-r", action="store_true", help="Search subdirectories")
    idx.add_argument("--skip-existing", action="store_true", help="Skip already indexed files")
    idx.add_argument("--no-diarization", action="store_true", help="Disable diarization")
    idx.add_argument("--no-events", action="store_true", help="Disable event detection")
    idx.add_argument("--language", type=str, default=None, help="Force language")
    idx.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── query ───
    q = subparsers.add_parser("query", help="Search indexed audio")
    q.add_argument("text", help="Search query text")
    q.add_argument("--speaker", type=str, default=None, help="Filter by speaker")
    q.add_argument("--events", type=str, default=None, help="Filter by events (comma-sep)")
    q.add_argument("--top-k", type=int, default=5, help="Max results")
    q.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── events ───
    ev = subparsers.add_parser("events", help="Search by audio events")
    ev.add_argument("filepath", help="Audio file path")
    ev.add_argument("--type", type=str, default=None, help="Event type filter")
    ev.add_argument("--top-k", type=int, default=10, help="Max results")
    ev.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── speakers ───
    sp = subparsers.add_parser("speakers", help="List speakers in a file")
    sp.add_argument("filepath", help="Audio file path")
    sp.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── timeline ───
    tl = subparsers.add_parser("timeline", help="Speaker timeline for a file")
    tl.add_argument("filepath", help="Audio file path")
    tl.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── register-speaker ───
    rs = subparsers.add_parser("register-speaker", help="Register a speaker")
    rs.add_argument("name", help="Speaker name")
    rs.add_argument("--audio", required=True, help="Reference audio file")
    rs.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── list-speakers ───
    ls = subparsers.add_parser("list-speakers", help="List registered speakers")
    ls.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── list ───
    li = subparsers.add_parser("list", help="List indexed files")
    li.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── stats ───
    st = subparsers.add_parser("stats", help="Show index statistics")
    st.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── remove ───
    rm = subparsers.add_parser("remove", help="Remove file from index")
    rm.add_argument("filepath", help="Audio file path")
    rm.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── export ───
    ex = subparsers.add_parser("export", help="Export indexed data")
    ex.add_argument("filepath", help="Audio file path")
    ex.add_argument("--format", type=str, default="json", help="Format: json/csv/srt/vtt/txt/notes")
    ex.add_argument("--output", "-o", type=str, default=None, help="Output file path")
    ex.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── listen ───
    lsn = subparsers.add_parser("listen", help="Start live listening")
    lsn.add_argument("--source", type=str, default="microphone", help="Audio source")
    lsn.add_argument("--on-keyword", type=str, default=None, help="Alert keywords (comma-sep)")
    lsn.add_argument("--on-event", type=str, default=None, help="Alert events (comma-sep)")
    lsn.add_argument("--db", type=str, default="sonarwise.db", help="Database path")

    # ─── config ───
    cfg = subparsers.add_parser("config", help="Show or set configuration")
    cfg.add_argument("--show", action="store_true", help="Show current config")
    cfg.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set config value")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        _run_command(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_command(args):
    from sonarwise.core.pipeline import SonarWise
    import os

    db = getattr(args, "db", "sonarwise.db")

    if args.command == "index":
        diarization = not getattr(args, "no_diarization", False)
        events = not getattr(args, "no_events", False)
        sw = SonarWise(diarization=diarization, events=events, db_path=db, verbose=True)

        if os.path.isdir(args.path):
            n = sw.index_folder(args.path, recursive=args.recursive, skip_existing=args.skip_existing)
        else:
            n = sw.index(args.path, skip_existing=args.skip_existing)
        print(f"Indexed {n} segments")

    elif args.command == "query":
        sw = SonarWise(db_path=db)
        event_list = args.events.split(",") if args.events else None
        results = sw.query(args.text, top_k=args.top_k, speaker=args.speaker, events=event_list)

        if not results:
            print("No results found.")
            return

        for i, r in enumerate(results, 1):
            ts = f"{ms_to_short(r.start_ms)}-{ms_to_short(r.end_ms)}"
            speaker = r.speaker_name or r.speaker_id or ""
            speaker_str = f" [{speaker}]" if speaker else ""
            print(f"{i}. [{ts}]{speaker_str} {r.transcript} (score: {r.score})")
            print(f"   File: {r.filepath}")

    elif args.command == "events":
        sw = SonarWise(db_path=db)
        if args.type:
            results = sw.query_events(args.type, top_k=args.top_k, filepath=args.filepath)
        else:
            # Show all events for the file
            segments = sw._store.get_segments_by_file(os.path.abspath(args.filepath))
            results = []
            for seg in segments:
                if seg.event_tags and seg.event_tags != ["speech"]:
                    from sonarwise.core.models import EventResult
                    results.append(EventResult(
                        filepath=seg.filepath,
                        start_ms=seg.start_ms,
                        end_ms=seg.end_ms,
                        event_tags=seg.event_tags,
                        event_confidence=seg.event_confidence,
                    ))

        for r in results:
            ts = f"{ms_to_short(r.start_ms)}-{ms_to_short(r.end_ms)}"
            tags = ", ".join(r.event_tags)
            print(f"[{ts}] {tags}")

    elif args.command == "speakers":
        sw = SonarWise(db_path=db)
        speakers = sw.get_speakers(args.filepath)
        for s in speakers:
            name = s.name or s.id
            dur = ms_to_short(s.duration_ms)
            print(f"  {name}: {dur} ({s.segment_count} segments)")

    elif args.command == "timeline":
        sw = SonarWise(db_path=db)
        timeline = sw.speaker_timeline(args.filepath)
        for speaker, ranges in timeline.items():
            parts = [f"{ms_to_short(s)}-{ms_to_short(e)}" for s, e in ranges]
            print(f"  {speaker}: {' | '.join(parts)}")

    elif args.command == "register-speaker":
        sw = SonarWise(diarization=True, db_path=db, verbose=True)
        sw.register_speaker(args.name, args.audio)
        print(f"Registered speaker: {args.name}")

    elif args.command == "list-speakers":
        sw = SonarWise(db_path=db)
        speakers = sw.list_registered_speakers()
        if not speakers:
            print("No registered speakers.")
        for s in speakers:
            print(f"  {s.name} (registered: {s.registered_at})")

    elif args.command == "list":
        sw = SonarWise(db_path=db)
        sources = sw.list_sources()
        if not sources:
            print("No indexed files.")
        for s in sources:
            dur = ms_to_short(s.duration_ms)
            print(f"  {s.filepath} ({s.segment_count} segments, {dur}, {s.speakers} speakers)")

    elif args.command == "stats":
        sw = SonarWise(db_path=db)
        s = sw.stats()
        print(f"Files:      {s['total_files']}")
        print(f"Segments:   {s['total_segments']}")
        print(f"Duration:   {ms_to_short(s['total_duration_ms'])}")
        print(f"Speakers:   {s['total_speakers']} detected, {s['registered_speakers']} registered")
        print(f"Languages:  {', '.join(s['unique_languages']) or 'none'}")
        print(f"DB size:    {s['db_size_mb']} MB")
        if s["event_distribution"]:
            print("Events:")
            for evt, count in sorted(s["event_distribution"].items(), key=lambda x: -x[1]):
                print(f"  {evt}: {count}")

    elif args.command == "remove":
        sw = SonarWise(db_path=db)
        n = sw.remove(args.filepath)
        print(f"Removed {n} segments")

    elif args.command == "export":
        sw = SonarWise(db_path=db)
        out = sw.export(args.filepath, format=args.format, output=args.output)
        print(f"Exported to {out}")

    elif args.command == "listen":
        sw = SonarWise(diarization=True, events=True, db_path=db, verbose=True, mode="live")

        @sw.on("transcript")
        def on_transcript(seg):
            speaker = seg.speaker_name or seg.speaker_id or ""
            prefix = f"[{speaker}] " if speaker else ""
            print(f"  {prefix}{seg.transcript}")

        if args.on_keyword:
            keywords = args.on_keyword.split(",")

            @sw.on("keyword", words=keywords)
            def on_kw(seg):
                print(f"  KEYWORD ALERT: {seg.transcript}")

        if args.on_event:
            events = args.on_event.split(",")

            @sw.on("sound_event", events=events)
            def on_evt(event):
                print(f"  EVENT: {event.event_tags} at {ms_to_short(event.start_ms)}")

        print(f"Listening on {args.source}... (Ctrl+C to stop)")
        sw.listen(source=args.source)

        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            sw.stop()
            print("\nStopped.")

    elif args.command == "config":
        from sonarwise.config import SonarWiseConfig

        if args.show:
            config = SonarWiseConfig.auto_load()
            for key, value in config.__dict__.items():
                print(f"  {key}: {value}")
        elif args.set:
            key, value = args.set
            config = SonarWiseConfig.auto_load()
            if hasattr(config, key):
                # Type coerce
                current = getattr(config, key)
                if isinstance(current, bool):
                    value = value.lower() in ("true", "1", "yes")
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                setattr(config, key, value)
                config.to_yaml("sonarwise.yaml")
                print(f"Set {key} = {value}")
            else:
                print(f"Unknown config key: {key}")


if __name__ == "__main__":
    main()
