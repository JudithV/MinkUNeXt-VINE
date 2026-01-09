# Miguel Hernández University of Elche
# Automation, Robotics and Computer Vision lab (ARCV)
# Judith Vilella Cantos
import pandas as pd
import matplotlib.pyplot as plt

"""
This file illustrates all the ground truth samples in a trajectory, colored by whether the latest saved evaluation 
was a correct match at each position (green) or not (red).
"""

# === 1. Charge results (can be saved in the pnv_evaluate.py file) ===
df = pd.read_csv("query_results_train.csv")

# === 2. Filters correct and incorrect matches ===
aciertos = df[df["correct@25"] == True]
fallos = df[df["correct@25"] == False]

# === 3. Plot ===
plt.figure(figsize=(10, 8))

# General trajectory (optional: grey line)
plt.plot(df["easting"], df["northing"], color='lightgray', linewidth=1, label='Trajectory')

# Correct matches → green
plt.scatter(aciertos["easting"], aciertos["northing"], 
            c='green', s=20, label='Correct (Top-25)', alpha=0.8)

# Incorrect → red
plt.scatter(fallos["easting"], fallos["northing"], 
            c='red', s=20, label='Incorrect (Top-25)', alpha=0.8)

plt.xlabel("Easting [m]")
plt.ylabel("Northing [m]")
plt.title("Spatial distribution of correct/incorrect recognitions (Top-25)")
plt.legend()
plt.axis('equal')
plt.grid(True)
#plt.show()
plt.savefig("results_predictions_25.png", dpi=300)

