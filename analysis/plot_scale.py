import matplotlib.pyplot as plt

# Data
pool_size = [8, 32, 64, 100]
germanic_pos = [72.1, 75.6, 77.1, 79.8]
african_ner = [69.2, 70.18, 70.88, 71.1]

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(pool_size, germanic_pos, marker='o', markersize=10, label='Germanic POS', color='blue', linewidth=2.5)
plt.plot(pool_size, african_ner, marker='o', markersize=10, label='African NER', color='green', linewidth=2.5)

# Adding titles and labels
#plt.title('Performance Metrics by Pool Size', fontsize=16)
plt.xlabel('Candidate Pool Size', fontsize=18, fontweight='bold')
plt.ylabel('Avg. F1', fontsize=18, fontweight='bold')
plt.xticks(pool_size, fontsize=12)  # Set x-ticks to be the pool sizes
plt.yticks(fontsize=12)
plt.ylim(68, 80)  # Set y-axis limits for better visualization
plt.grid(True)
plt.legend(loc='lower right', fontsize=16)
# Save the plot as a PNG file
plt.savefig('SSP_scale.png')
# Show the plot
plt.show()
