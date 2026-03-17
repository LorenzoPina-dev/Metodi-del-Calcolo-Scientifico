import json
import matplotlib.pyplot as plt

# 1️⃣ Leggi i risultati JSON
with open("benchmark_results.json", "r") as f:
    data = json.load(f)

# 2️⃣ Ordina le dimensioni delle matrici
sizes = sorted([int(x) for x in data.keys()])

# 3️⃣ Estrai i metodi disponibili dal primo elemento
methods = list(data[str(sizes[0])].keys())

# 4️⃣ Crea un dizionario che mappa metodo -> lista di tempi
timings = {method: [] for method in methods}
for n in sizes:
    for method in methods:
        timings[method].append(data[str(n)][method]/1000)

# 5️⃣ Plot con matplotlib
plt.figure(figsize=(12, 8))

for method in methods:
    plt.plot(sizes, timings[method], marker="o", label=method)

plt.title("Matrix Benchmark (Performance vs Dimensione)")
plt.xlabel("Dimensione matrice n (n x n)")
plt.ylabel("Tempo (s)")
plt.grid(True)
plt.legend()

# 6️⃣ Mostra grafico
plt.tight_layout()
plt.show()