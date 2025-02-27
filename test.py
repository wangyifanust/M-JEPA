import re
import matplotlib.pyplot as plt

# File path to the .out file
file_path = '/home/yifwang/M-JEPA/slurm_out3/200291.out'

# Initialize lists to store epochs and averaged loss values
epochs = []
avg_loss_values = []

# Regular expression to match lines with averaged stats and capture the value within parentheses
averaged_stats_pattern = re.compile(r'Averaged stats: .*?loss: [\d\.]+ \(([\d\.]+)\)')

# Read the .out file and extract averaged loss values within parentheses
with open(file_path, 'r') as file:
    epoch_counter = 0
    for line in file:
        match = averaged_stats_pattern.search(line)
        if match:
            avg_loss = float(match.group(1))  # Use the value within parentheses
            
            epoch_counter += 1
            if epoch_counter %1==0: 
                epochs.append(epoch_counter)
                avg_loss_values.append(avg_loss)

# Plotting the averaged loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, avg_loss_values, marker='.', linestyle='-', color='r')
plt.xlabel('Epoch')
plt.ylabel('Averaged Loss')
plt.title('Averaged Loss Progression over Epochs')
plt.grid(True)
output_path = './averaged_loss_progression.png'
plt.savefig(output_path)

# Show the figure as well
plt.show()
