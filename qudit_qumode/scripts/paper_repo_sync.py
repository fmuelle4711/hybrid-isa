"""Sync the paper repository to exactly the manifest.

Reads results/paper_manifest.txt and makes
Documents/hybrid_qudit_oscillator_isa hold exactly those files under
qudit_qumode/ and hybridlane/.  Anything else under those two directories
is removed.  Top-level LICENSE, README.md, requirements-frozen.txt and
tools/ are left alone.

DOES NOT COMMIT AND DOES NOT PUSH.  Prints what it changed.

    python scripts/paper_repo_sync.py --dry-run    show the plan only
    python scripts/paper_repo_sync.py              apply it
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HL_SRC = ROOT.parent / "hybridlane"
PAPER = Path("C:/Users/mhakimi/Documents/hybrid_qudit_oscillator_isa")
MANIFEST = ROOT / "results" / "paper_manifest.txt"

KEEP_TOP = {"LICENSE", "README.md", "REPRODUCE.md", "requirements-frozen.txt",
            "tools", ".git", ".gitignore"}


def read_manifest():
    """path -> (source path, reason).  Skips comments and MISSING entries."""
    out = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if "*** MISSING ***" in line:
            continue
        rel, _, reason = line.partition("  ")
        rel, reason = rel.strip(), reason.strip()
        if not rel:
            continue
        src = None
        if "[copied from " in reason:
            src = reason.split("[copied from ", 1)[1].split("]")[0].strip()
            reason = reason.split("  [copied from")[0]
        if rel.startswith("hybridlane/"):
            source = HL_SRC / rel[len("hybridlane/"):]
        else:
            source = ROOT / (src or rel)
        out[rel] = (source, reason)
    return out


def dest_for(rel):
    if rel.startswith("hybridlane/"):
        return PAPER / rel
    return PAPER / "qudit_qumode" / rel


def main():
    dry = "--dry-run" in sys.argv
    man = read_manifest()
    print(f"manifest entries: {len(man)}")

    wanted = {dest_for(r).resolve() for r in man}

    # ------------------------------------------------------------ remove --
    removed = []
    for top in ("qudit_qumode", "hybridlane"):
        d = PAPER / top
        if not d.exists():
            continue
        for p in sorted(d.rglob("*"), reverse=True):
            if ".git" in p.parts:
                continue
            if p.is_file() and p.resolve() not in wanted:
                removed.append(p.relative_to(PAPER).as_posix())
                if not dry:
                    p.unlink()
            elif p.is_dir() and not dry and not any(p.iterdir()):
                p.rmdir()

    # ------------------------------------------------------------- copy ---
    copied, failed = [], []
    for rel, (src, _) in sorted(man.items()):
        dst = dest_for(rel)
        if not src.exists():
            failed.append((rel, f"source missing: {src}"))
            continue
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied.append(rel)

    print(f"copied : {len(copied)}")
    print(f"removed: {len(removed)}")
    if removed:
        for r in removed[:40]:
            print(f"    - {r}")
        if len(removed) > 40:
            print(f"    ... and {len(removed)-40} more")
    if failed:
        print(f"FAILED : {len(failed)}")
        for rel, why in failed:
            print(f"    ! {rel}  {why}")
    if dry:
        print("\n(dry run; nothing written)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
