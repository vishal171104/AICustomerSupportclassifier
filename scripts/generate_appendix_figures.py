import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

os.makedirs('figures', exist_ok=True)

# 1. Category Distribution
labels = ['Technical', 'Billing', 'Account']
counts = [683, 214, 185]

plt.figure(figsize=(8, 5))
sns.barplot(x=labels, y=counts, palette='Blues_r')
plt.title('Category Class Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
for i, v in enumerate(counts):
    plt.text(i, v + 10, str(v), color='black', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/category_distribution.png', dpi=300)
plt.close()

# 2. Copy the existing specific reports to the placeholders
shutil.copy('reports/cat_confusion_matrix.png', 'figures/category_confusion_matrix.png')
shutil.copy('reports/pri_confusion_matrix.png', 'figures/priority_confusion_matrix.png')
shutil.copy('reports/lc_svm.png', 'figures/svm_learning_curve.png')
shutil.copy('reports/lc_ensemble.png', 'figures/ensemble_learning_curve.png')

print("All appendix figures successfully copied/generated in figures/")
