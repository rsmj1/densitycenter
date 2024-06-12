import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator



costs = [np.inf, 100, 95, 90, 50, 45, 43, 42, 41.5, 41.2, 41, 40.8]
costs = costs[1:]
print(costs)
k_values = list(range(2, len(costs) + 1))  # Assuming k starts at 1 and increments by 1
# Use KneeLocator to find the elbow point
kn = KneeLocator(k_values, costs, curve='convex', direction='decreasing')

# Output the optimal k value
print(f"The optimal k value is: {kn.knee}")
# Index for x-axis
time = range(1, len(costs)+1)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(time, costs, marker='o', color='b', linestyle='-')
plt.title('Costs Over Time')
plt.xlabel('Time')
plt.ylabel('Cost')
plt.xticks(time)
plt.grid(True)
plt.show()