import pandas as pd
import matplotlib.pyplot as plt

# Carga el CSV
df = pd.read_csv("vmd/pergola/run2_06_p/gps.csv")

# Si tus datos están en lat/lon, puedes convertirlos a UTM o trabajar directamente en 2D
x = df["x"] if "x" in df.columns else df["longitude"]
y = df["y"] if "y" in df.columns else df["latitude"]
labels = df["segment"]
color = ['blue', 'yellow', 'red', 'cyan', 'magenta', 'green', 'black', 'orange', 'brown', 'pink', 'purple', 'gold', 'lime', 'teal', 'lavender', 'maroon', 'turquoise', 'olive', 'coral']

colores_hex = ['#FF5733', '#33FF57', '#5733FF', '#FF33C7', '#33FFAA', '#57FF33', '#AA33FF', '#FF3357', '#33AAFF', '#33FF33', '#FFAA33', '#57FFAA', '#AAFF33', '#FF33AA', '#33AA57', '#5733AA', '#AA5733', '#57AA33', '#AA3357', '#3357AA', '#7FFFC7', '#7FC7FF', '#FF7FC7', '#FFC77F', '#C77FFF', '#FFC733', '#FF33C7', '#33FF7F', '#7FFF33', '#337FFF', '#7FFF33', '#33C7FF', '#C7FF33', '#7FFFC7', '#7FC7FF', '#FF7FC7', '#FFC77F', '#C77FFF', '#FFC733', '#FF33C7', '#33FF7F']


# Visualización
plt.figure(figsize=(10, 10))
for i in range(len(df)):
    plt.plot(x[i], y[i], marker='o', markersize=5, linestyle='None', color=color[labels[i]])

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Segmentación del viñedo por etiquetas")
#plt.show()
plt.savefig('segmentation.png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
