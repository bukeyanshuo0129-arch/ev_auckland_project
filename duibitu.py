import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read Excel file
df = pd.read_excel('Comparison.xlsx', sheet_name='Sheet1')

# Split data for two days
day1 = df.iloc[0:24]  # First 24 rows for day 1
day2 = df.iloc[24:48]  # Next 24 rows for day 2

# Get charging station columns (exclude Hour column)
stations = df.columns[1:]

# Set font to support English and proper minus sign
plt.rcParams['font.sans-serif'] = ['Arial']  # Use Arial font
plt.rcParams['axes.unicode_minus'] = False  # Display minus sign correctly

# Create 2x3 subplot layout
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Charging Station Load Comparison', fontsize=16)

# Plot curves for each charging station
for i, station in enumerate(stations):
    row, col = i // 3, i % 3
    ax = axes[row, col]
    
    # Plot data for two days
    ax.plot(day1['Hour'], day1[station], label='Predicted', marker='o', markersize=3)
    ax.plot(day2['Hour'], day2[station], label='Actual', marker='s', markersize=3)
    
    # Set title and labels
    ax.set_title(f'{station} Load Comparison')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Load')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_ylim(bottom=0)
    
# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save figure
plt.savefig('charging_station_load_comparison.png', dpi=300, bbox_inches='tight')

# Show figure
plt.show()
