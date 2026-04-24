import json
import matplotlib.pyplot as plt
import os
import numpy as np

with open('results/baseline_div_stats.json') as f:
    stats = json.load(f)

models = list(stats.keys())
means = [stats[m]["mean"] for m in models]
stds  = [stats[m]["std"] for m in models]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, means, yerr=stds, capsize=5, color=['blue', 'orange', 'green', 'red', 'purple'])
plt.yscale('log')
plt.ylabel(r'$\varepsilon_{\mathrm{div}} = \|\mathbf{G} \mathbf{a}\|_2$')
plt.title('Figure 9: Constraint Enforcement Comparison')
plt.xticks(rotation=15)
plt.grid(axis='y', ls='--', alpha=0.5)

for bar, mean in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, mean*1.1, f'{mean:.2e}', ha='center')

os.makedirs('results/figures', exist_ok=True)
plt.tight_layout()
plt.savefig('results/figures/fig9_constraint_enforcement.png', dpi=400, bbox_inches='tight')
print("✅ Figure 9 saved: results/figures/fig9_constraint_enforcement.png")
plt.show()
