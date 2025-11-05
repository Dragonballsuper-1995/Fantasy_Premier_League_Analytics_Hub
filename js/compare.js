// Compare page logic extracted from inline script
let chartInstance1 = null, chartInstance2 = null;
let currentPlayer1 = null, currentPlayer2 = null;
let ALL_PLAYER_DATA = [], ALL_MODELS = [];
let activeSuggestion = { 1: -1, 2: -1 };
let filteredPlayersStore = { 1: [], 2: [] };

const modelSelector = document.getElementById('modelSelector');
const searchInput1 = document.getElementById('searchInput1'), searchSuggestions1 = document.getElementById('searchSuggestions1');
const loadingStatus = document.getElementById('loadingStatus');
const results1 = document.getElementById('results1'), noResults1 = document.getElementById('noResults1');
const playerCard1 = document.getElementById('playerCard1'), chartTitle1 = document.getElementById('chartTitle1');
const chartCanvas1 = document.getElementById('pointsChart1'), upcomingFixturesList1 = document.getElementById('upcomingFixturesList1');
const searchInput2 = document.getElementById('searchInput2'), searchSuggestions2 = document.getElementById('searchSuggestions2');
const results2 = document.getElementById('results2'), noResults2 = document.getElementById('noResults2');
const playerCard2 = document.getElementById('playerCard2'), chartTitle2 = document.getElementById('chartTitle2');
const chartCanvas2 = document.getElementById('pointsChart2'), upcomingFixturesList2 = document.getElementById('upcomingFixturesList2');
const totalPlayersEl = document.getElementById('totalPlayers'), availablePlayersEl = document.getElementById('availablePlayers'), injuredPlayersEl = document.getElementById('injuredPlayers');

document.addEventListener('themechange', function() {
	if (currentPlayer1) updatePlayerDisplay(1, currentPlayer1);
	if (currentPlayer2) updatePlayerDisplay(2, currentPlayer2);
});

async function loadData() {
	try {
		const response = await fetch(`predictions.json?v=${new Date().getTime()}`);
		if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
		const responseText = await response.text();
		if (responseText.includes('NaN')) {
			throw new Error("Invalid JSON: 'NaN' found in predictions.json. Please re-run 'python main.py'.");
		}
		ALL_PLAYER_DATA = JSON.parse(responseText);
		if (!Array.isArray(ALL_PLAYER_DATA) || ALL_PLAYER_DATA.length === 0) {
			throw new Error("'predictions.json' is empty or not a valid array.");
		}
		const firstPlayerWithData = ALL_PLAYER_DATA.find(p => p.upcoming_predictions && p.upcoming_predictions.length > 0 && p.upcoming_predictions[0].predictions);
		if (firstPlayerWithData) {
			ALL_MODELS = Object.keys(firstPlayerWithData.upcoming_predictions[0].predictions);
			modelSelector.innerHTML = ALL_MODELS.map(model => {
				const isRecommended = model.includes('XGBoost') || model.includes('Random Forest');
				return `<option value="${model}">${model}${isRecommended ? ' (Rec.)' : ''}</option>`
			}).join('');
			if (ALL_MODELS.includes('XGBoost')) modelSelector.value = 'XGBoost';
		}
		searchInput1.disabled = false; searchInput1.placeholder = "e.g., Haaland, Saka, Palmer...";
		searchInput2.disabled = false; searchInput2.placeholder = "e.g., Foden, Son, Salah...";
		loadingStatus.textContent = `Successfully loaded ${ALL_PLAYER_DATA.length} players.`;
		calculateGlobalStats();
		checkURLParams();
	} catch (error) {
		console.error("Error loading predictions.json:", error);
		loadingStatus.innerHTML = `<strong class="text-red-600 dark:text-red-400">Error: Could not load data.</strong><br>${error.message}`;
	}
}
document.addEventListener('DOMContentLoaded', loadData);

function checkURLParams() {
	const urlParams = new URLSearchParams(window.location.search);
	const player1Id = urlParams.get('player1');
	if (player1Id && ALL_PLAYER_DATA.length > 0) {
		const player = ALL_PLAYER_DATA.find(p => p.id == player1Id);
		if (player) displayPlayerData(player, 1);
	}
}

function calculateGlobalStats() {
	const totalPlayers = ALL_PLAYER_DATA.length;
	const availablePlayers = ALL_PLAYER_DATA.filter(p => p.chance_of_playing === 100 || p.chance_of_playing == null).length;
	const injuredPlayers = totalPlayers - availablePlayers;
	totalPlayersEl.textContent = totalPlayers;
	availablePlayersEl.textContent = availablePlayers;
	injuredPlayersEl.textContent = injuredPlayers;
}

modelSelector.addEventListener('change', () => {
	if (currentPlayer1) updatePlayerDisplay(1, currentPlayer1);
	if (currentPlayer2) updatePlayerDisplay(2, currentPlayer2);
});

searchInput1.addEventListener('input', (e) => handleSearch(e.target.value, 1));
searchInput2.addEventListener('input', (e) => handleSearch(e.target.value, 2));

function handleSearch(query, slot) {
	const noResultsEl = (slot === 1) ? noResults1 : noResults2;
	activeSuggestion[slot] = -1;
	query = query.toLowerCase().trim();
	if (query.length < 2) { clearSuggestions(slot); hideResults(slot); return; }
	const filteredPlayers = ALL_PLAYER_DATA.filter(p => p.web_name && p.web_name.toLowerCase().includes(query)).sort((a,b) => a.web_name.localeCompare(b.web_name));
	filteredPlayersStore[slot] = filteredPlayers;
	if (filteredPlayers.length > 0) { showSuggestions(filteredPlayers, slot); noResultsEl.classList.add('hidden'); }
	else { clearSuggestions(slot); hideResults(slot); noResultsEl.classList.remove('hidden'); }
}

function showSuggestions(players, slot) {
	const suggestionsEl = (slot === 1) ? searchSuggestions1 : searchSuggestions2;
	suggestionsEl.innerHTML = '';
	suggestionsEl.classList.remove('hidden');
	players.slice(0, 5).forEach((player, index) => {
		const suggestionItem = document.createElement('div');
		suggestionItem.className = 'p-3 hover:bg-stone-100 dark:hover:bg-stone-600 cursor-pointer text-sm';
		suggestionItem.textContent = `${player.web_name} (${player.position} / ${player.team || 'N/A'}) - £${(player.cost || 0).toFixed(1)}m`;
		suggestionItem.addEventListener('click', () => displayPlayerData(player, slot));
		suggestionItem.addEventListener('mouseover', () => {
			removeActive(suggestionsEl.childNodes);
			suggestionItem.classList.add('suggestion-active');
			activeSuggestion[slot] = index;
		});
		suggestionsEl.appendChild(suggestionItem);
	});
}

function clearSuggestions(slot) {
	const suggestionsEl = (slot === 1) ? searchSuggestions1 : searchSuggestions2;
	suggestionsEl.classList.add('hidden');
	suggestionsEl.innerHTML = '';
	activeSuggestion[slot] = -1;
}

document.addEventListener('click', (e) => {
	if (searchInput1 && searchSuggestions1 && !searchInput1.contains(e.target) && !searchSuggestions1.contains(e.target)) clearSuggestions(1);
	if (searchInput2 && searchSuggestions2 && !searchInput2.contains(e.target) && !searchSuggestions2.contains(e.target)) clearSuggestions(2);
});

function displayPlayerData(player, slot) {
	if (slot === 1) { currentPlayer1 = player; searchInput1.value = player.web_name; }
	else { currentPlayer2 = player; searchInput2.value = player.web_name; }
	clearSuggestions(slot);
	updatePlayerDisplay(slot, player);
}

function updatePlayerDisplay(slot, player) {
	const resultsEl = (slot === 1) ? results1 : results2;
	const noResultsEl = (slot === 1) ? noResults1 : noResults2;
	const cardEl = (slot === 1) ? playerCard1 : playerCard2;
	const chartTitleEl = (slot === 1) ? chartTitle1 : chartTitle2;
	const fixturesEl = (slot === 1) ? upcomingFixturesList1 : upcomingFixturesList2;
	const chartCanvas = (slot === 1) ? chartCanvas1 : chartCanvas2;
	const selectedModel = modelSelector.value;

	const chance = player.chance_of_playing;
	let statusText = '<p class="text-sm font-medium text-green-600 dark:text-green-500 mt-1">Available</p>';
	if (player.news && chance < 100) {
		const color = (chance === 0) ? 'red' : 'yellow';
		statusText = `<p class="text-sm font-medium text-${color}-600 dark:text-${color}-500 mt-1">${player.news} (${chance}% chance)</p>`;
	} else if (chance === 100 && player.news) {
		statusText = `<p class="text-sm font-medium text-green-600 dark:text-green-500 mt-1">${player.news}</p>`;
	}

	cardEl.innerHTML = `
			<div class="flex items-center space-x-4 mb-6">
					<div class="w-16 h-16 rounded-full bg-amber-600 flex items-center justify-center text-white text-3xl font-bold uppercase">${player.web_name.charAt(0)}</div>
					<div>
							<h2 class="text-3xl font-bold">${player.web_name}</h2>
							<p class="text-xl text-stone-600 dark:text-stone-400">${player.team || 'Unknown'} - <span class="font-bold">${player.position || 'UNK'}</span></p>
							${statusText}
					</div>
			</div>
			<div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
					<div class="bg-stone-100 dark:bg-stone-700 p-4 rounded-lg text-center">
							<p class="text-sm font-medium text-stone-500 dark:text-stone-400 uppercase">Cost</p>
							<p class="text-4xl font-bold">£${(player.cost || 0).toFixed(1)}m</p>
					</div>
					<div class="bg-stone-100 dark:bg-stone-700 p-4 rounded-lg text-center">
							<p class="text-sm font-medium text-stone-500 dark:text-stone-400 uppercase">Prev GW</p>
							<p class="text-4xl font-bold">${player.prev_gw_points}</p>
					</div>
					<div class="bg-amber-100 dark:bg-stone-700 p-4 rounded-lg text-center">
							<p class="text-sm font-medium text-amber-700 dark:text-amber-300 uppercase">Next GW</p>
							<p class="text-4xl font-bold text-amber-600 dark:text-amber-400">${(player.upcoming_predictions?.[0]?.predictions?.[selectedModel] ?? 0).toFixed(1)}</p>
							<p class="text-xs text-stone-500 dark:text-stone-400 mt-1 font-semibold">vs. ${player.next_opponent || 'N/A'}</p>
					</div>
			</div>
	`;

	const labels = player.last_5_gw_labels || ['?','?','?','?','?'];
	chartTitleEl.textContent = `Points Trend (${labels[0]} - ${labels[4]})`;
	renderPointsChart(player.last_5_gw_points, labels, player.last_5_gw_opponents, chartCanvas, slot);
	renderUpcomingFixtures(fixturesEl, player.upcoming_predictions, selectedModel);
	resultsEl.classList.remove('hidden');
	noResultsEl.classList.add('hidden');
}

function hideResults(slot) {
	document.getElementById(slot === 1 ? 'results1' : 'results2').classList.add('hidden');
}

function renderUpcomingFixtures(element, fixtures, selectedModel) {
	element.innerHTML = '';
	if (!fixtures || fixtures.length === 0) {
		element.innerHTML = '<p class="text-stone-500 dark:text-stone-400 italic text-sm">No upcoming fixtures found.</p>';
		return;
	}
	fixtures.forEach(fixture => {
		const predictedScore = (fixture.predictions?.[selectedModel] ?? 0.0);
		const listItem = document.createElement('div');
		listItem.className = 'flex justify-between items-center py-2 border-b border-stone-200 dark:border-stone-700 last:border-b-0 text-sm';
		listItem.innerHTML = `
				<div>
						<span class="font-semibold text-stone-700 dark:text-stone-300">GW ${fixture.gw}:</span>
						<span class="ml-2 text-stone-600 dark:text-stone-400">${fixture.opponent || 'Unknown'}</span>
				</div>
				<span class="font-bold text-lg text-amber-600 dark:text-amber-500">${predictedScore.toFixed(1)}</span>
		`;
		element.appendChild(listItem);
	});
}

function renderPointsChart(data, labels, opponents, canvasEl, slot) {
	let chartInstance = (slot === 1) ? chartInstance1 : chartInstance2;
	if (chartInstance) chartInstance.destroy();

	const safeData = (Array.isArray(data) && data.length === 5) ? data : [0,0,0,0,0];
	const safeLabels = (Array.isArray(labels) && labels.length === 5) ? labels : ['?', '?', '?', '?', '?'];
	const safeOpponents = (Array.isArray(opponents) && opponents.length === 5) ? opponents : ['-', '-', '-', '-', '-'];
	const isDark = document.documentElement.classList.contains('dark');
	const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';
	const labelColor = isDark ? '#a8a29e' : '#57534e';
	const pointColor = (slot === 1) ? '#d97706' : '#16a34a';
	const lineColor = (slot === 1) ? '#f59e0b' : '#22c55e';
	const fillColor = isDark
		? (slot === 1 ? 'rgba(245, 158, 11, 0.3)' : 'rgba(34, 197, 94, 0.3)')
		: (slot === 1 ? 'rgba(253, 230, 138, 0.5)' : 'rgba(187, 247, 208, 0.5)');

	const ctx = canvasEl.getContext('2d');
	chartInstance = new Chart(ctx, {
		type: 'line',
		data: {
			labels: safeLabels,
			datasets: [{
				label: 'Points', data: safeData, borderColor: lineColor, backgroundColor: fillColor,
				tension: 0.3, borderWidth: 2, pointRadius: 4, pointBackgroundColor: pointColor,
				pointBorderColor: isDark ? '#1c1917' : '#fffbeb', pointHoverRadius: 6, fill: true,
				pointOpponents: safeOpponents
			}]
		},
		options: {
			responsive: true, maintainAspectRatio: false,
			scales: {
				y: { beginAtZero: true, suggestedMax: Math.max(...safeData, 5) + 2, ticks: { color: labelColor, precision: 0 }, grid: { color: gridColor, drawBorder: false } },
				x: { ticks: { color: labelColor }, grid: { display: false } }
			},
			plugins: {
				legend: { display: false },
				tooltip: {
					backgroundColor: isDark ? '#292524' : '#ffffff', titleColor: isDark ? '#f5f5f4' : '#1c1917',
					bodyColor: isDark ? '#d6d3d1' : '#44403c', borderColor: isDark ? '#44403c' : '#e7e5e4',
					borderWidth: 1, padding: 10, boxPadding: 4,
					callbacks: {
						title: (tooltipItems) => tooltipItems[0].label || '',
						label: (context) => {
							const points = context.parsed.y;
							const opponent = context.dataset.pointOpponents[context.dataIndex] || '-';
							return ` Points: ${points} (vs ${opponent})`;
						}
					}
				}
			},
			interaction: { intersect: false, mode: 'index' }
		}
	});
	if (slot === 1) chartInstance1 = chartInstance; else chartInstance2 = chartInstance;
}

function addKeydownListener(inputEl, suggestionsEl, slot) {
	inputEl.addEventListener('keydown', function(e) {
		const items = suggestionsEl.getElementsByTagName('div');
		if (items.length === 0) return;
		if (e.key === 'ArrowDown') { e.preventDefault(); activeSuggestion[slot]++; if (activeSuggestion[slot] >= items.length) activeSuggestion[slot] = 0; addActive(items, slot); }
		else if (e.key === 'ArrowUp') { e.preventDefault(); activeSuggestion[slot]--; if (activeSuggestion[slot] < 0) activeSuggestion[slot] = items.length - 1; addActive(items, slot); }
		else if (e.key === 'Enter') { e.preventDefault(); if (activeSuggestion[slot] > -1) { items[activeSuggestion[slot]].click(); } }
	});
}
function addActive(items, slot) {
	removeActive(items);
	if (!items[activeSuggestion[slot]]) return;
	items[activeSuggestion[slot]].classList.add('suggestion-active');
	items[activeSuggestion[slot]].scrollIntoView({ block: 'nearest' });
}
function removeActive(items) {
	for (let i = 0; i < items.length; i++) items[i].classList.remove('suggestion-active');
}
addKeydownListener(searchInput1, searchSuggestions1, 1);
addKeydownListener(searchInput2, searchSuggestions2, 2);
