#!/usr/bin/env python3
"""
Fetch Official NASA/USGS Mars DEMs

Downloads the DEM GeoTIFFs referenced in the problem statement:
- Mars MOLA Global DEM (463m)
- HRSC/MOLA Blended Global DEM (200m)
- MSL Gale Crater DEM (1m)

Files are saved under data/dem/ with original filenames.
"""

import os
import sys
import requests
from pathlib import Path
from typing import List

DEMS = {
    "mola": "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif",
    "gale": "https://planetarymaps.usgs.gov/mosaic/Mars/MSL/MSL_Gale_DEM_Mosaic_1m_v3.tif",
    "hrsc": "https://planetarymaps.usgs.gov/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif",
}


def download(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    out_path = out_dir / filename

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f" Already downloaded: {out_path}")
        return out_path

    print(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = 1024 * 1024
        downloaded = 0
        with open(out_path, "wb") as f:
            for data in r.iter_content(chunk_size=chunk):
                if not data:
                    continue
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded * 100 // total
                    print(f"  {pct}%", end="\r", flush=True)
    print(f" Saved to: {out_path}")
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch official Mars DEM GeoTIFFs")
    parser.add_argument(
        "--dems",
        nargs="*",
        default=["mola", "hrsc", "gale"],
        choices=list(DEMS.keys()),
        help="Which DEMs to download"
    )
    parser.add_argument("--out", default="data/dem", help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out)

    for key in args.dems:
        url = DEMS[key]
        try:
            download(url, out_dir)
        except Exception as e:
            print(f" Failed to download {key}: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
