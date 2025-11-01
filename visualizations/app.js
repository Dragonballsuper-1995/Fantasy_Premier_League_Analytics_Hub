const seasons = [
  "2016-17", "2017-18", "2018-19", "2019-20", 
  "2020-21", "2021-22", "2022-23", "2023-24", 
  "2024-25", "2025-26"
];

const baseURL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/";

const playerSelect = document.getElementById("player");
const chartDiv = document.getElementById("chart");

let combinedData = [];

async function loadAllData() {
  for (const season of seasons) {
    try {
      const url = `${baseURL}${season}/cleaned_players.csv`;
      const response = await fetch(url);
      const text = await response.text();
      const rows = text.split("\n").slice(1);
      
      for (const row of rows) {
        const cols = row.split(",");
        if (cols.length > 10) {
          combinedData.push({
            name: cols[1],
            gw: parseInt(cols[3]),
            total_points: parseFloat(cols[20]),
            season: season
          });
        }
      }
      console.log(`Loaded ${season}`);
    } catch (e) {
      console.warn(`Failed to load ${season}`, e);
    }
  }

  populatePlayerDropdown();
}

function populatePlayerDropdown() {
  const playerNames = [...new Set(combinedData.map(d => d.name))].sort();
  playerSelect.innerHTML = playerNames.map(name => `<option>${name}</option>`).join("");
  updateGraph(playerSelect.value);
}

function updateGraph(playerName) {
  const playerData = combinedData.filter(d => d.name === playerName);
  if (!playerData.length) return;

  const seasonsGrouped = {};
  playerData.forEach(d => {
    if (!seasonsGrouped[d.season]) seasonsGrouped[d.season] = [];
    seasonsGrouped[d.season].push(d);
  });

  const traces = Object.keys(seasonsGrouped).map(season => {
    const data = seasonsGrouped[season].sort((a, b) => a.gw - b.gw);
    return {
      x: data.map(d => d.gw),
      y: data.map(d => d.total_points),
      mode: "lines+markers",
      name: season
    };
  });

  const layout = {
    title: `${playerName} - Total Points (All Seasons)`,
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#161b22",
    font: { color: "#c9d1d9" },
    xaxis: { title: "Gameweek" },
    yaxis: { title: "Total Points" },
  };

  Plotly.newPlot(chartDiv, traces, layout);
}

playerSelect.addEventListener("change", () => updateGraph(playerSelect.value));

loadAllData();
