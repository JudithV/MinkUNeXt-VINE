import pandas as pd
import matplotlib.pyplot as plt

# === 1. Carga los resultados ===
# Cambia el nombre según tu archivo CSV
df = pd.read_csv("query_results_train.csv")

# === 2. Filtra aciertos y fallos ===
aciertos = df[df["correct@25"] == True]
fallos = df[df["correct@25"] == False]

# === 3. Plot ===
plt.figure(figsize=(10, 8))

# Trayectoria general (opcional: línea gris)
plt.plot(df["easting"], df["northing"], color='lightgray', linewidth=1, label='Trajectory')

# Aciertos → verde
plt.scatter(aciertos["easting"], aciertos["northing"], 
            c='green', s=20, label='Correct (Top-25)', alpha=0.8)

# Fallos → rojo
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

