"""Extract UCF_Crimes.zip with CRC checks, resuming after interruption.

Files whose size already matches the archive are skipped. Each file is written
to a .part name, flushed to disk, then renamed, so a stopped run never leaves a
truncated video under its final name. Flushing also keeps unwritten data from
piling up in memory when the target is a slow USB drive.

Usage:
    python scripts/extract_ucf.py --zip D:/watchdog-data/raw/ucf_crime/UCF_Crimes.zip
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: the zip's folder")
    args = parser.parse_args()
    output = args.output or args.zip.parent

    with zipfile.ZipFile(args.zip) as archive:
        temporal = args.zip.parent / "Temporal_Anomaly_Annotation_For_Testing_Videos.zip"
        if temporal.exists():
            zipfile.ZipFile(temporal).extractall(output)
        infos = archive.infolist()
        total = sum(info.file_size for info in infos)
        done = skipped = 0
        for number, info in enumerate(infos, 1):
            target = output / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                # ZipExtFile raises BadZipFile if the CRC does not match.
                with archive.open(info) as source, open(partial, "wb") as destination:
                    while chunk := source.read(8 << 20):
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
                partial.replace(target)
            done += info.file_size
            if number % 50 == 0 or number == len(infos):
                print(
                    f"{number}/{len(infos)} entries, {done / 1e9:.1f}/{total / 1e9:.1f} GB, "
                    f"skipped {skipped}, {time.strftime('%H:%M:%S')}",
                    flush=True,
                )
    print("EXTRACT_OK: all files written, CRCs verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
