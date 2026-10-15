"""Segment tables, splits and labels for extraction (signdeg/extract.py) and store building.

A segment table is a TSV with columns vid, yid, start, end: `yid` names the video file <yid>.mp4 and every
row is one item of the output store. A clip is one whole-video segment.

  durations   video dir -> durations.tsv (yid, seconds)
  source      pre-training source split: YouTube-ASL videos without OpenASL evaluation material, plus
              non-overlapping YouTube-SL-25 ASL videos that replace the removed duration
              -> source_segments.tsv (one whole-video segment per source video)
  how2sign    How2Sign sentence clips -> h2s_segments.tsv, h2s_split.json
  openasl     OpenASL table -> openasl_segments.tsv, openasl_split.json
  aslcitizen  ASL Citizen release -> aslc_segments.tsv, aslc_split.json, aslc_labels.json
  clips       generic clip list (CSV: video, split[, label][, text]) for Sem-Lex, WLASL2000, FLEURS-ASL,
              ASL-STEM-Wiki, ... -> <name>_segments.tsv, <name>_split.json[, _labels.json][, _text.tsv]
"""
import argparse
import csv
import json
from pathlib import Path

WHOLE = ("00:00:00.000", "99:00:00.000")      # a segment spanning the whole video


def write_segments(path: Path, rows):
    path.write_text("vid\tyid\tstart\tend\n" + "".join("\t".join(r) + "\n" for r in rows))


def durations(a):
    import cv2
    rows = []
    for p in sorted(Path(a.video_dir).glob("*.mp4")):
        cap = cv2.VideoCapture(str(p))
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        rows.append(f"{p.stem}\t{n / fps:.2f}\n")
    Path(a.out).write_text("yid\tseconds\n" + "".join(rows))
    print(len(rows), "videos")


def source(a):
    dur = {r["yid"]: float(r["seconds"]) for r in csv.DictReader(open(a.durations), delimiter="\t")}
    yt = {r["yid"] for r in csv.DictReader(open(a.ytasl_table), delimiter="\t")}
    evaluation = {r["yid"] for r in csv.DictReader(open(a.openasl_table), delimiter="\t")
                  if r.get("split", "") not in ("train", "")}
    kept = sorted(y for y in yt if y not in evaluation and y in dur)
    removed = sum(dur.get(y, 0.0) for y in yt if y in evaluation)
    added, extra = 0.0, []
    for y in [l.strip() for l in open(a.ytsl25_list) if l.strip()]:
        if added >= removed:
            break
        if y in yt or y in evaluation or y not in dur:
            continue
        extra.append(y)
        added += dur[y]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_segments(out / "source_segments.tsv", [(y, y) + WHOLE for y in kept + extra])
    hours = (sum(dur[y] for y in kept) + added) / 3600
    print(f"YouTube-ASL kept {len(kept)} videos, removed {removed / 3600:.1f} h with OpenASL evaluation clips; "
          f"YouTube-SL-25 added {len(extra)} videos ({added / 3600:.1f} h); total {hours:.1f} h")


def how2sign(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = {}
    for sp in ("train", "val", "test"):
        with open(Path(a.csv_dir) / f"how2sign_{sp}.csv", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                split[r["SENTENCE_NAME"]] = sp
    write_segments(out / "h2s_segments.tsv", [(v, v) + WHOLE for v in sorted(split)])
    (out / "h2s_split.json").write_text(json.dumps(split))
    print(len(split), "sentences")


def openasl(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, split = [], {}
    for r in csv.DictReader(open(a.table), delimiter="\t"):
        rows.append((r["vid"], r["yid"], r["start"], r["end"]))
        split[r["vid"]] = {"dev": "val", "validation": "val"}.get(r["split"], r["split"])
    write_segments(out / "openasl_segments.tsv", rows)
    (out / "openasl_split.json").write_text(json.dumps(split))
    print(len(rows), "clips")


def aslcitizen(a):
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split, lab = {}, {}
    for sp in ("train", "val", "test"):
        with open(root / "splits" / f"{sp}.csv", newline="") as f:
            for r in csv.DictReader(f):
                v = r["Video file"][:-4]
                split[v], lab[v] = sp, r["Gloss"]
    write_segments(out / "aslc_segments.tsv", [(v, v) + WHOLE for v in split])
    (out / "aslc_split.json").write_text(json.dumps(split))
    (out / "aslc_labels.json").write_text(json.dumps(lab))
    print(len(split), "videos", len(set(lab.values())), "glosses")


def clips(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv)))
    vid = lambda r: Path(r["video"]).stem  # noqa: E731
    write_segments(out / f"{a.name}_segments.tsv", [(vid(r), vid(r)) + WHOLE for r in rows])
    (out / f"{a.name}_split.json").write_text(json.dumps({vid(r): r["split"] for r in rows}))
    if "label" in rows[0]:
        (out / f"{a.name}_labels.json").write_text(json.dumps({vid(r): r["label"] for r in rows}))
    if "text" in rows[0]:
        (out / f"{a.name}_text.tsv").write_text("vid\ttext\tsplit\n" + "".join(
            f"{vid(r)}\t{r['text']}\t{r['split']}\n" for r in rows))
    print(len(rows), "clips")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("durations")
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("source")
    p.add_argument("--ytasl-table", required=True, help="YouTube-ASL caption table (column yid)")
    p.add_argument("--openasl-table", required=True, help="OpenASL table (columns yid, split)")
    p.add_argument("--ytsl25-list", required=True, help="YouTube-SL-25 ASL video ids, one per line")
    p.add_argument("--durations", required=True, help="durations.tsv of all candidate videos")
    p.add_argument("--out", required=True)
    p = sub.add_parser("how2sign")
    p.add_argument("--csv-dir", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("openasl")
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("aslcitizen")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("clips")
    p.add_argument("--csv", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"durations": durations, "source": source, "how2sign": how2sign, "openasl": openasl,
     "aslcitizen": aslcitizen, "clips": clips}[a.cmd](a)


if __name__ == "__main__":
    main()
