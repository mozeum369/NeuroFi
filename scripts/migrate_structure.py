#!/usr/bin/env python3
"""
NeuroFi folder migration script.

Features:
- Dry-run by default (preview only)
- Git-aware moves (uses `git mv` when --git is passed)
- Auto-backup copied files to .migrate_backup_YYYYmmdd_HHMMSS/
- Creates required target directories if missing
- Merges data/indexs -> data/indexes
- Moves select files/folders to new structure
- Optionally creates symlinks at old locations (--symlink-old-paths)
- Cleans up empty source directories

Usage:
  python scripts/migrate_structure.py              # Dry run
  python scripts/migrate_structure.py --execute --git
  python scripts/migrate_structure.py --execute --git --symlink-old-paths

Run from repo root.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (assumes scripts/ file)
TS = time.strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = ROOT / f".migrate_backup_{TS}"

def is_git_repo() -> bool:
    try:
        subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def run_git_mv(src: Path, dst: Path) -> bool:
    try:
        subprocess.run(["git", "-C", str(ROOT), "mv", str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))],
                       check=True)
        return True
    except Exception:
        return False

def backup_file(src: Path, dry_run: bool):
    rel = src.relative_to(ROOT)
    backup_path = BACKUP_DIR / rel
    if dry_run:
        print(f"[DRY] Backup: {rel} -> {backup_path}")
        return
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, backup_path)

def safe_move(src: Path, dst: Path, *, execute: bool, use_git: bool, create_parents: bool = True) -> None:
    if not src.exists():
        print(f"[SKIP] Source missing: {src.relative_to(ROOT)}")
        return
    if create_parents:
        if not execute:
            print(f"[DRY] Ensure dir: {dst.parent.relative_to(ROOT)}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        # Avoid overwriting—backup src and then replace if same file is desired
        print(f"[WARN] Destination exists: {dst.relative_to(ROOT)}. Backing up source before replacing.")
        backup_file(src, dry_run=(not execute))
        if execute:
            if dst.is_file():
                dst_backup = dst.with_suffix(dst.suffix + f".pre_migrate_{TS}.bak")
                dst.rename(dst_backup)
                print(f"[INFO] Existing dest moved to backup: {dst_backup.relative_to(ROOT)}")

    if not execute:
        print(f"[DRY] Move: {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        return

    if use_git and is_git_repo():
        # Try git mv; fallback to shutil if it fails (e.g., across filesystems)
        if run_git_mv(src, dst):
            print(f"[GIT] mv {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
            return

    # Regular move
    shutil.move(str(src), str(dst))
    print(f"[MV ] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

def create_symlink(old_path: Path, new_path: Path, *, execute: bool):
    if not execute:
        print(f"[DRY] Symlink: {old_path.relative_to(ROOT)} -> {new_path.relative_to(ROOT)}")
        return
    old_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if old_path.exists() or old_path.is_symlink():
            # do not overwrite; place a .link instead
            link_alt = old_path.with_suffix(old_path.suffix + ".link")
            if link_alt.exists() or link_alt.is_symlink():
                link_alt = link_alt.with_suffix(link_alt.suffix + ".dup")
            old_path = link_alt
        old_path.symlink_to(new_path)
        print(f"[LNK] {old_path.relative_to(ROOT)} -> {new_path.relative_to(ROOT)}")
    except OSError as e:
        print(f"[WARN] Symlink failed ({old_path} -> {new_path}): {e}")

def remove_if_empty(path: Path, *, execute: bool):
    try:
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            if not execute:
                print(f"[DRY] rmdir: {path.relative_to(ROOT)}")
            else:
                path.rmdir()
                print(f"[RM ] dir {path.relative_to(ROOT)}")
    except Exception as e:
        print(f"[WARN] Could not remove {path}: {e}")

def collect_document_jsons(base: Path) -> list[Path]:
    # data/documents/document/*.json -> data/documents/
    src_dir = base / "data" / "documents" / "document"
    if src_dir.exists():
        return [p for p in src_dir.glob("*.json") if p.is_file()]
    return []

def main():
    parser = argparse.ArgumentParser(description="NeuroFi folder migration")
    parser.add_argument("--execute", action="store_true", help="Apply changes (default is dry run)")
    parser.add_argument("--git", action="store_true", help="Use git mv when possible")
    parser.add_argument("--symlink-old-paths", action="store_true", help="Create symlinks at old locations to new ones")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backup copies")
    args = parser.parse_args()

    execute = args.execute
    use_git = args.git
    make_symlinks = args.symlink_old_paths
    do_backup = not args.no_backup

    print(f"== NeuroFi Migration (dry_run={not execute}, git={use_git}, symlinks={make_symlinks}) ==")

    # Sanity checks
    if not (ROOT / "README.md").exists():
        print("[WARN] README.md not found at repo root; ensure you are running from the NeuroFi repo.")
    # Prepare backup dir
    if do_backup and execute:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Backup dir: {BACKUP_DIR.relative_to(ROOT)}")

    # Ensure target directories
    required_dirs = [
        ROOT / "tests",
        ROOT / "deploy",
        ROOT / "data" / "indexes",
        ROOT / "data" / "documents",
        ROOT / ".cache" / "ai_core",
        ROOT / "src" / "bot",
    ]
    for d in required_dirs:
        if not execute:
            print(f"[DRY] Ensure dir: {d.relative_to(ROOT)}")
        else:
            d.mkdir(parents=True, exist_ok=True)

    # 1) Merge data/indexs -> data/indexes
    old_indexes = ROOT / "data" / "indexs"
    new_indexes = ROOT / "data" / "indexes"
    if old_indexes.exists():
        for item in old_indexes.glob("*"):
            dst = new_indexes / item.name
            if do_backup and item.is_file():
                backup_file(item, dry_run=(not execute))
            safe_move(item, dst, execute=execute, use_git=use_git, create_parents=True)
        remove_if_empty(old_indexes, execute=execute)

    # 2) Move utils/render.yaml -> deploy/render.yaml
    safe_move(ROOT / "utils" / "render.yaml", ROOT / "deploy" / "render.yaml", execute=execute, use_git=use_git)

    # 3) Move scripts/Cryptobot.py -> src/bot/Cryptobot.py
    safe_move(ROOT / "scripts" / "Cryptobot.py", ROOT / "src" / "bot" / "Cryptobot.py", execute=execute, use_git=use_git)

    # 4) Move coinbase/test_coinbase_jwt.py -> tests/test_coinbase_jwt.py
    safe_move(ROOT / "coinbase" / "test_coinbase_jwt.py", ROOT / "tests" / "test_coinbase_jwt.py", execute=execute, use_git=use_git)
    remove_if_empty(ROOT / "coinbase", execute=execute)

    # 5) Flatten data/documents/document/*.json -> data/documents/
    for doc in collect_document_jsons(ROOT):
        dst = ROOT / "data" / "documents" / doc.name
        if do_backup:
            backup_file(doc, dry_run=(not execute))
        safe_move(doc, dst, execute=execute, use_git=use_git)
    remove_if_empty(ROOT / "data" / "documents" / "document", execute=execute)

    # 6) Move ai_core/ai_core.tmp -> .cache/ai_core/ai_core.tmp
    safe_move(ROOT / "ai_core" / "ai_core.tmp", ROOT / ".cache" / "ai_core" / "ai_core.tmp", execute=execute, use_git=False)

    # Optional symlinks at old paths
    if make_symlinks:
        # For the three primary moves
        links = [
            (ROOT / "utils" / "render.yaml", ROOT / "deploy" / "render.yaml"),
            (ROOT / "scripts" / "Cryptobot.py", ROOT / "src" / "bot" / "Cryptobot.py"),
            (ROOT / "coinbase" / "test_coinbase_jwt.py", ROOT / "tests" / "test_coinbase_jwt.py"),
            (ROOT / "data" / "indexs", ROOT / "data" / "indexes"),
            (ROOT / "data" / "documents" / "document", ROOT / "data" / "documents"),
            (ROOT / "ai_core" / "ai_core.tmp", ROOT / ".cache" / "ai_core" / "ai_core.tmp"),
        ]
        for oldp, newp in links:
            if newp.exists():
                create_symlink(oldp, newp, execute=execute)

    print("== Migration plan complete ==")
    if not execute:
        print("Dry run only. Re-run with --execute to apply.")
    else:
        print("Applied changes.")
        print("Reminder: review imports or scripts that referenced moved paths (e.g., scripts/Cryptobot.py).")
        print("Consider adding '.cache/' to your .gitignore if not already present.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
