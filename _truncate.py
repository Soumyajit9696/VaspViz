import sys
path = r'c:\Users\SOUMYAJIT\Desktop\vaspviz_dev - anti\New_anti_vaspviz_1may\widgets.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
# Keep only lines 1-1413 (index 0-1412) + one trailing newline
with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines[:1413])
    f.write('\n')
print(f"Truncated to {1413} lines + newline")
