import matplotlib.pyplot as plt

# Data
noise_prob = [0, 0.2, 0.4, 0.6, 0.8, 1]
Got = [21.8, 32.5, 62, 77.7, 78.5, 80.7]
Fo = [48.3, 75.3, 85.9, 89.7, 91.5, 93.5]
Gsw = [40.5, 58.6, 68.5, 83.2, 86.7, 89.9]

# Horizontal lines values
Fo_stage1 = 81.3
Got_stage1 = 66.5
Gsw_stage1 = 82.3

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(noise_prob, Fo, marker='o', label='Fo', color='orange', linewidth=3.5)
plt.plot(noise_prob, Got, marker='o', label='Got', color='blue', linewidth=3.5)
plt.plot(noise_prob, Gsw, marker='o', label='Gsw', color='green', linewidth=3.5)

# Adding dashed horizontal lines
plt.axhline(Fo_stage1, color='orange', linestyle='--', linewidth=3, label='Fo Stage 1')
plt.axhline(Got_stage1, color='blue', linestyle='--', linewidth=3, label='Got Stage 1')
plt.axhline(Gsw_stage1, color='green', linestyle='--', linewidth=3, label='Gsw Stage 1')

# Labels and title
#plt.title('Noise Probability vs Performance Metrics')
plt.xlabel('1-x (x=Label noise)', fontsize=18, fontweight='bold')
plt.ylabel('F1', fontsize=18, fontweight='bold')

plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlim(0, 1)
plt.legend(fontsize=16)
plt.grid()

# Save the plot as a PNG file
plt.savefig('noise_gpt4.png')

# Show the plot
plt.show()
