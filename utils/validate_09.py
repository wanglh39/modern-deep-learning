"""验证 09_tokenization.ipynb 的所有 code cell"""
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

nb_path = "notebooks/09_tokenization.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

namespace = {
    'np': np, 'plt': plt, 'matplotlib': matplotlib,
    're': re, 'Counter': Counter, 'defaultdict': defaultdict,
    'warnings': warnings,
}

code_cells = [c for c in nb.cells if c.cell_type == 'code']
print(f"共 {len(code_cells)} 个 code cell, 开始验证...\n")

for i, cell in enumerate(code_cells, 1):
    try:
        exec(cell.source, namespace)
        print(f"  Cell {i:2d}/{len(code_cells)} ✅")
    except Exception as e:
        print(f"  Cell {i:2d}/{len(code_cells)} ❌ {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        break
else:
    print("\n✅ 所有 code cell 验证通过!")