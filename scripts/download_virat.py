"""Download VIRAT Ground 2.0 videos that have DIVA activity annotations.

Videos come from Kitware's public Girder server. Downloads resume after
interruption and are checked against the server's file sizes.

Usage:
    python scripts/download_virat.py --annotations D:/watchdog-data/raw/virat/diva_annotations
        --output D:/watchdog-data/raw/virat/videos
"""

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://data.kitware.com/api/v1"
VIDEOS_FOLDER = "56f581ce8d777f753209ca43"  # VIRAT Ground Dataset / videos_original
CHUNK = 1 << 20


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def annotated_names(annotations):
    return sorted({path.name.split(".")[0] for path in annotations.glob("*/*.activities.yml")})


def download(item, output, retries=10):
    target = output / item["name"]
    partial = target.with_name(target.name + ".part")
    if target.exists() and target.stat().st_size == item["size"]:
        return item["name"], "exists"
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            f"{API}/item/{item['_id']}/download", headers={"Range": f"bytes={offset}-"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    offset = 0  # server ignored the range; restart the file
                with open(partial, "ab" if offset else "wb") as handle:
                    while chunk := response.read(CHUNK):
                        handle.write(chunk)
            if partial.stat().st_size == item["size"]:
                partial.replace(target)
                return item["name"], "downloaded"
        except OSError as error:
            print(f"{item['name']}: attempt {attempt} failed ({error})", flush=True)
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"{item['name']}: incomplete after {retries} attempts")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    wanted = set(annotated_names(args.annotations))
    items = get_json(f"{API}/item?folderId={VIDEOS_FOLDER}&limit=1000")
    selected = [item for item in items if item["name"].rsplit(".", 1)[0] in wanted]
    missing = sorted(wanted - {item["name"].rsplit(".", 1)[0] for item in selected})
    total = sum(item["size"] for item in selected)
    print(f"{len(selected)} videos, {total / 1e9:.1f} GB; {len(missing)} annotated videos not published")
    args.output.mkdir(parents=True, exist_ok=True)

    failures = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download, item, args.output) for item in selected]
        for count, future in enumerate(as_completed(futures), 1):
            try:
                name, status = future.result()
                print(f"[{count}/{len(selected)}] {name}: {status}", flush=True)
            except RuntimeError as error:
                failures += 1
                print(f"[{count}/{len(selected)}] {error}", flush=True)
    (args.output / "missing_videos.txt").write_text("\n".join(missing) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
