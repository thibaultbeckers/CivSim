import os
import subprocess
from pathlib import Path

# === Settings ===
current_file = Path(__file__).name
root_dir = Path(__file__).parent

# Read .gitignore if it exists
gitignore_path = root_dir / ".gitignore"
ignored_paths = set()

if gitignore_path.exists():
    with open(gitignore_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                ignored_paths.add(stripped.rstrip("/"))

# Collect all Python files recursively, skipping ignored
py_files = []

for path in root_dir.rglob("*.py"):
    rel_path = path.relative_to(root_dir)
    if (
        rel_path.name == current_file  # skip this script
        or any(str(rel_path).startswith(ignored) for ignored in ignored_paths)
        or path.stat().st_size == 0  # skip empty files
    ):
        continue

    # Skip files with only whitespace
    try:
        if not path.read_text(encoding="utf-8").strip():
            continue
    except Exception as e:
        print(f"Skipping {path} (read error): {e}")
        continue

    py_files.append(path)

# Write to output
combined_content = []

for path in sorted(py_files):
    rel_path = path.relative_to(root_dir)
    content = path.read_text(encoding="utf-8")
    header = f"### {rel_path} ###\n"
    combined_content.append(header + content + "\n")

# Copy to clipboard
try:
    subprocess.run("pbcopy", input="".join(combined_content), text=True, check=True)
    print("Combined Python code copied to clipboard.")
except Exception as e:
    print(f"Clipboard copy failed: {e}")
