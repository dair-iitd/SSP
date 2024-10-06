import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['SSP', 'w/o label', 'w/o conf', 'w/o both']
ner_values = [52.4, 50.8, 50.0, 51.0]
pos_values = [62.33333333, 61.33333333, 46.33333333, 47.66666667]

# Create a figure and a set of subplots
fig, ax1 = plt.subplots()

# Set positions for the NER bars
x = np.arange(len(labels))

# Plot NER values
ax1.bar(x - 0.2, ner_values, width=0.4, label='NER', color='blue', alpha=0.6)

# Create a second y-axis for POS
ax2 = ax1.twinx()
ax2.bar(x + 0.2, pos_values, width=0.4, label='POS', color='orange', alpha=0.6)

# Adding labels and titles
ax1.set_xlabel('Categories')
ax1.set_ylabel('NER Values', color='blue')
ax2.set_ylabel('POS Values', color='orange')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# Title and grid
plt.title('NER and POS Values Comparison')
ax1.grid(axis='y')

# Adding legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()

plt.savefig('prec.png')
plt.show()
