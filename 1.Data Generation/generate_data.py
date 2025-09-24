import matplotlib.pyplot as plt
import numpy as np
import random
import os

#1. CONFIGURATION
NUMBER_T = 100      # Number of triangle images
NUMBER_C = 100      # Number of circle images
FOLDER = "DataTC"   # Folder to save images

# Create folder if not exists
os.makedirs(FOLDER, exist_ok=True)


#2. TRIANGLE GENERATION
# Generates 3 random points forming a valid triangle.
def generate_3_points(x_range, y_range):
    while True:
        points = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(3)]
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        # Calculate area using determinant method
        area = abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0)
        # Ensure non-collinear points
        if area > 0:  
            return np.array(points).T

#3. CIRCLE GENERATION
# Generates random circle parameters: center + radius.
def generate_circle():
    center_x = random.uniform(1.5, 3.5)
    center_y = random.uniform(1.5, 3.5)
    radius = random.uniform(0.5, 1.5)
    return center_x, center_y, radius

#4. FINAL LOOPS
for i in range(NUMBER_T):
    x, y = generate_3_points((0, 5), (0, 5))

    # Draw triangle with random fill and edge colors
    plt.clf()
    fill = (random.random(), random.random(), random.random())
    edge = (random.random(), random.random(), random.random())
    plt.fill(x, y, color=fill, edgecolor=edge)

    # Remove axes for clean image
    plt.axis("off")

    # Save image
    plt.savefig(f"{FOLDER}/t/t{i}.png", bbox_inches='tight', pad_inches=0, dpi=300)

for i in range(NUMBER_C):
    cx, cy, r = generate_circle()

    # Create circle points using parametric equation
    theta = np.linspace(0, 2 * np.pi, 100)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    # Draw circle with random fill and edge colors
    plt.clf()
    fill = (random.random(), random.random(), random.random())
    edge = (random.random(), random.random(), random.random())
    plt.fill(x, y, color=fill, edgecolor=edge)

    # 🔹 FIXED: Keep the same coordinate limits for every image
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.gca().set_aspect('equal', adjustable='box')  # Keep perfect circle proportions

    # Remove axes for clean image
    plt.axis("off")

    # Save image
    plt.savefig(f"{FOLDER}/c/c{i}.png", bbox_inches='tight', pad_inches=0, dpi=300)

