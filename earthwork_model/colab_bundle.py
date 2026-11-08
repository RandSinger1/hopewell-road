"""
colab_bundle.py — Package a dataset + the training code for Google Colab.

    python -m earthwork_model.colab_bundle --dataset data/earthworks/dataset/v1

Writes to --out (default parameters.EW_COLAB_DIR = G:/My Drive/khthon, the Google
Drive for desktop folder Colab mounts as /content/drive/MyDrive/khthon — so the
files reach Drive without a manual upload):
    khthon_code.zip         parameters.py, earthwork_model/, earthwork_processor/{features,augment}.py
                            — everything train.py imports; no GDAL / PDAL needed (small, re-upload
                            whenever the code changes)
    dataset_<name>.zip      manifest.json + chips/ as dataset/<name>/ (stored, not deflated: the
                            .npz chips are already compressed).  Skipped while it is newer than
                            the dataset's manifest.json (build_dataset writes that last), so an
                            unchanged 3.6 GB dataset is not re-uploaded; --force rebuilds it
    colab_train.ipynb       earthwork_model/colab_train.ipynb with DATASET set to this dataset

Drive for desktop uploads them in the background (watch its tray icon before
starting Colab).  Open colab_train.ipynb from Drive in Colab, pick a GPU runtime
and run all cells.
Checkpoints are written to the Drive folder, so a disconnected session resumes
where it stopped (train.py --resume).  Afterwards copy models/<name>/best.pt
back to data/earthworks/models/<name>/.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import parameters

ROOT = Path(__file__).resolve().parent.parent
CODE_FILES = ["parameters.py",
              "earthwork_model/__init__.py", "earthwork_model/dataset.py",
              "earthwork_model/model.py", "earthwork_model/train.py",
              "earthwork_processor/__init__.py", "earthwork_processor/features.py",
              "earthwork_processor/augment.py"]
NOTEBOOK = ROOT / "earthwork_model" / "colab_train.ipynb"


def bundle_code(out: Path) -> Path:
    path = out / "khthon_code.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE_FILES:
            z.write(ROOT / f, f)
    return path


def dataset_zip_current(dataset: Path, out: Path) -> bool:
    """The zip exists and was written after the dataset was last (re)built."""
    path = out / f"dataset_{dataset.name}.zip"
    return path.exists() and path.stat().st_mtime > (dataset / "manifest.json").stat().st_mtime


def bundle_dataset(dataset: Path, out: Path) -> Path:
    manifest = json.loads((dataset / "manifest.json").read_text())
    path = out / f"dataset_{dataset.name}.zip"
    tmp = path.with_name(path.name + ".tmp")
    prefix = f"dataset/{dataset.name}"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        z.write(dataset / "manifest.json", f"{prefix}/manifest.json")
        for i, chip in enumerate(manifest["chips"]):
            z.write(dataset / chip["file"], f"{prefix}/{chip['file']}")
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(manifest['chips'])} chips")
    tmp.replace(path)
    return path


def write_notebook(out: Path, dataset_name: str) -> Path:
    """Copy colab_train.ipynb with its DATASET setting pointing at this bundle's dataset."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        cell["source"] = [re.sub(r'^DATASET = "[^"]*"', f'DATASET = "{dataset_name}"', line)
                          for line in cell["source"]]
    path = out / NOTEBOOK.name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=parameters.EW_ROOT / "dataset" / "default")
    ap.add_argument("--out", type=Path, default=parameters.EW_COLAB_DIR,
                    help="Default: parameters.EW_COLAB_DIR (Google Drive for desktop)")
    ap.add_argument("--code-only", action="store_true", help="Skip the dataset zip (code changed, data did not)")
    ap.add_argument("--force", action="store_true", help="Rebuild the dataset zip even if it is current")
    args = ap.parse_args()
    if not (args.dataset / "manifest.json").exists():
        raise SystemExit(f"No manifest.json in {args.dataset} - run earthwork_processor.build_dataset first")
    if not args.out.parent.exists():
        raise SystemExit(f"{args.out.parent} does not exist - is Google Drive for desktop running and "
                         f"signed in?  (Or pass --out / change EW_COLAB_DIR.)")
    args.out.mkdir(parents=True, exist_ok=True)
    written = [bundle_code(args.out)]
    if args.code_only:
        pass
    elif dataset_zip_current(args.dataset, args.out) and not args.force:
        print(f"dataset_{args.dataset.name}.zip is newer than the dataset - not re-uploaded (--force to rebuild)")
    else:
        print(f"Zipping {args.dataset} ...")
        written.append(bundle_dataset(args.dataset, args.out))
    written.append(write_notebook(args.out, args.dataset.name))
    for p in written:
        print(f"  {p}  ({p.stat().st_size / 1e6:,.1f} MB)")
    print(f"Google Drive for desktop uploads these in the background - wait for its tray icon to finish, then "
          f"open {NOTEBOOK.name} from Drive in Colab (DATASET = \"{args.dataset.name}\") and run all cells.")


if __name__ == "__main__":
    main()
