import matplotlib.pyplot as plt

# Read the data
distances = []
errors = []

with open('prueba2.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        dist = float(parts[0].split(':')[1].strip().replace(' m', ''))
        err = float(parts[1].split(':')[1].strip().replace(' m', ''))
        distances.append(dist)
        errors.append(err)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(distances, errors, alpha=0.6, s=20)
plt.xlabel('Distance (m)')
plt.ylabel('Error (m)')
plt.title('Distance vs Error')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()