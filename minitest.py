import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator



def find_elbow_point(costs):
    # Convert to numpy array
    costs = np.array(costs)
    
    # Calculate the difference between consecutive costs
    diff = np.diff(costs)
    
    # Find the index of the maximum second difference (i.e., the maximum curvature)
    second_diff = np.diff(diff)
    elbow_index = np.argmax(second_diff) + 1  # +1 to adjust for the second difference index
    
    return elbow_index

costs = [100, 80, 60, 50, 45, 43, 42, 41.5, 41.2, 41, 40.8]
k_values = list(range(1, len(costs) + 1))  # Assuming k starts at 1 and increments by 1
# Use KneeLocator to find the elbow point
kn = KneeLocator(k_values, costs, curve='convex', direction='decreasing')

# Output the optimal k value
print(f"The optimal k value is: {kn.knee}")
print("k for elbow:", find_elbow_point(costs))
# Index for x-axis
time = range(len(costs))

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(time, costs, marker='o', color='b', linestyle='-')
plt.title('Costs Over Time')
plt.xlabel('Time')
plt.ylabel('Cost')
plt.xticks(time)
plt.grid(True)
plt.show()