import fs from "fs";
import path from "path";
import { writeFileSync } from "fs";
import Plotly from "plotly.js";
import Plotly from "plotly.js-node"; // libreria compatibile Node.js per grafici PNG

// Path al file JSON dei risultati
const resultsFile = path.join(__dirname, "benchmark_results.json");
if (!fs.existsSync(resultsFile)) {
    console.error("File benchmark_results.json non trovato!");
    process.exit(1);
}

const raw = fs.readFileSync(resultsFile, "utf-8");
const data = JSON.parse(raw);

const sizes = Object.keys(data)
    .map(k => parseInt(k))
    .sort((a, b) => a - b);

// Metodi da graficare
const methods = ["LUP","LU","LU_TotalPivot","Solve","Determinant","Inverse","Sparse_LUP"];

const traces: Plotly.ScatterData[] = methods.map(m => ({
    x: sizes,
    y: sizes.map(s => data[s][m]),
    type: "scatter",
    mode: "lines+markers",
    name: m
} as Plotly.ScatterData));

const layout: Partial<Plotly.Layout> = {
    title: { text: "Matrix Benchmark" },
    xaxis: { title: { text: 'Matrix size (n x n)' } },
    yaxis: { title: { text: 'Time (ms)' }, type: 'log' as const }
};

// Genera immagine PNG
const imgPath = path.join(__dirname, "benchmark_results.png");

Plotly.toImage({ data: traces, layout: layout, format: "png", width: 1200, height: 800 })
    .then((imgData: string) => {
        // imgData è base64
        const base64Data = imgData.replace(/^data:image\/png;base64,/, "");
        fs.writeFileSync(imgPath, base64Data, "base64");
        console.log(`Grafico salvato in: ${imgPath}`);
    })
    .catch(err => console.error(err));