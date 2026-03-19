// Matrix.visualize.ts
import fs from 'fs';
import { JSDOM } from 'jsdom';

const resultsPath = './matrix_full_benchmark_results.json';
const results = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));

// Creiamo una pagina HTML base
const dom = new JSDOM(`<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Matrix Benchmark Visualization</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; }
    canvas { margin: 30px 0; }
    table { border-collapse: collapse; margin-bottom: 30px; }
    th, td { border: 1px solid #333; padding: 4px 8px; text-align: right; }
    th { background: #eee; }
  </style>
</head>
<body>
  <h1>Matrix Benchmark Results</h1>
  <div id="tables"></div>
  <div id="charts"></div>
</body>
</html>
`);

const document = dom.window.document;

// --- CREA TABELLE ---
const tablesDiv = document.getElementById('tables')!;

for (const sizeStr of Object.keys(results)) {
    const size = parseInt(sizeStr);
    const table = document.createElement('table');
    table.innerHTML = `<thead><tr><th>Metodo</th><th>${size}x${size} (ms)</th></tr></thead>`;
    const tbody = document.createElement('tbody');

    for (const [method, time] of Object.entries(results[sizeStr])) {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${method}</td><td>${time}</td>`;
        tbody.appendChild(row);
    }

    table.appendChild(tbody);
    tablesDiv.appendChild(table);
}

// --- CREA GRAFICI ---
const chartsDiv = document.getElementById('charts')!;
const canvas = document.createElement('canvas');
canvas.id = 'benchmarkChart';
canvas.width = 1000;
canvas.height = 600;
chartsDiv.appendChild(canvas);

const sizesLabels = Object.keys(results);
const methodNames = Object.keys(results[sizesLabels[0]]);

const datasets = methodNames.map(method => {
    return {
        label: method,
        data: sizesLabels.map(size => results[size][method]),
        borderWidth: 1,
        fill: false,
        backgroundColor: `rgba(${Math.floor(Math.random()*200)}, ${Math.floor(Math.random()*200)}, ${Math.floor(Math.random()*200)}, 0.6)`,
        borderColor: `rgba(${Math.floor(Math.random()*200)}, ${Math.floor(Math.random()*200)}, ${Math.floor(Math.random()*200)}, 1)`
    };
});

const script = document.createElement('script');
script.textContent = `
const ctx = document.getElementById('benchmarkChart').getContext('2d');
new Chart(ctx, {
    type: 'line',
    data: {
        labels: ${JSON.stringify(sizesLabels)},
        datasets: ${JSON.stringify(datasets)}
    },
    options: {
        responsive: true,
        plugins: { legend: { position: 'bottom' } },
        scales: { y: { beginAtZero: true, title: { display: true, text: 'Tempo (ms)' } },
                  x: { title: { display: true, text: 'Dimensione matrice' } } }
    }
});
`;
document.body.appendChild(script);

// salva come file HTML
fs.writeFileSync('matrix_benchmark_visual.html', dom.serialize());
console.log('File HTML generato: matrix_benchmark_visual.html');