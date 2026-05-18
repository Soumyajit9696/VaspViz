"""Remove duplicate method block from widgets.py (keep lines 1..1467 only)."""
path = "widgets.py"
lines = open(path, encoding="utf-8").readlines()
print(f"Original: {len(lines)} lines")
# Keep up to and including the _redraw method closing blank line (index 1466 = line 1467)
keep = lines[:1467]
open(path, "w", encoding="utf-8").writelines(keep)
print(f"Trimmed to: {len(keep)} lines")
