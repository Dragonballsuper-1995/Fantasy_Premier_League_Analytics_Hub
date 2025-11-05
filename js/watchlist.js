// Watchlist page logic extracted from inline script
let ALL_PLAYER_DATA = [], ALL_MODELS = [];

const modelSelector = document.getElementById('modelSelector');
const watchlistSelector = document.getElementById('watchlistSelector');
const watchlistTitle = document.getElementById('watchlistTitle');
const watchlistBox = document.getElementById('watchlistBox');
const loadingStatus = document.getElementById('loadingStatus');
const totalPlayersEl = document.getElementById('totalPlayers');
const availablePlayersEl = document.getElementById('availablePlayers');
const injuredPlayersEl = document.getElementById('injuredPlayers');

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
		loadingStatus.textContent = `Successfully loaded ${ALL_PLAYER_DATA.length} players.`;
		calculateGlobalStats();
		generateWatchlists();
	} catch (error) {
		console.error("Error loading predictions.json:", error);
		loadingStatus.innerHTML = `<strong class="text-red-600 dark:text-red-400">Error: Could not load data.</strong><br>${error.message}`;
	}
}
document.addEventListener('DOMContentLoaded', loadData);

function calculateGlobalStats() {
	const totalPlayers = ALL_PLAYER_DATA.length;
	const availablePlayers = ALL_PLAYER_DATA.filter(p => p.chance_of_playing === 100 || p.chance_of_playing == null).length;
	const injuredPlayers = totalPlayers - availablePlayers;
	totalPlayersEl.textContent = totalPlayers;
	availablePlayersEl.textContent = availablePlayers;
	injuredPlayersEl.textContent = injuredPlayers;
}

modelSelector.addEventListener('change', generateWatchlists);
watchlistSelector.addEventListener('change', generateWatchlists);

function generateWatchlists() {
	if (ALL_PLAYER_DATA.length === 0) return;
	const selectedModel = modelSelector.value;
	const selectedWatchlist = watchlistSelector.value;
	let filteredList = [], title = '';

	if (selectedWatchlist === 'VALUE') {
		filteredList = ALL_PLAYER_DATA.filter(p => p.cost > 0 && p.upcoming_predictions && p.upcoming_predictions.length > 0);
		title = 'Top 5 Value Players (All)';
	} else {
		filteredList = ALL_PLAYER_DATA.filter(p => p.position === selectedWatchlist && p.upcoming_predictions && p.upcoming_predictions.length > 0);
		title = `Top 5 ${watchlistSelector.options[watchlistSelector.selectedIndex].text}`;
	}

	const sortedList = filteredList.map(player => {
		const nextGwPred = player.upcoming_predictions[0]?.predictions?.[selectedModel] ?? 0;
		let score = nextGwPred;
		if (selectedWatchlist === 'VALUE' && player.cost > 0) score = (nextGwPred / player.cost);
		return { ...player, score };
	}).sort((a, b) => b.score - a.score).slice(0, 5);

	watchlistTitle.textContent = title;
	watchlistBox.innerHTML = '';
	if (sortedList.length === 0) {
		watchlistBox.innerHTML = '<p class="text-stone-500 dark:text-stone-400">No players found for this category.</p>';
		return;
	}

	sortedList.forEach((player, index) => {
		const item = document.createElement('div');
		item.className = 'flex items-center justify-between p-3 rounded-lg hover:bg-amber-50 dark:hover:bg-stone-700 cursor-pointer transition-colors';
		item.onclick = () => { window.location.href = `compare.html?player1=${player.id}`; };
		const scoreLabel = selectedWatchlist === 'VALUE' ? 'Value' : 'xPts';
		item.innerHTML = `
			<div class="flex items-center space-x-3">
				<span class="font-bold text-lg text-stone-400 dark:text-stone-500">${index + 1}</span>
				<div class="w-8 h-8 rounded-full bg-stone-300 dark:bg-stone-600 flex items-center justify-center text-sm font-bold uppercase">${player.web_name.slice(0, 2)}</div>
				<div>
					<p class="font-semibold">${player.web_name}</p>
					<p class="text-xs text-stone-500 dark:text-stone-400">${player.position} / ${player.team} - £${(player.cost || 0).toFixed(1)}m</p>
				</div>
			</div>
			<div class="text-right">
				<p class="text-lg font-bold text-amber-600 dark:text-amber-500">${player.score.toFixed(2)}</p>
				<p class="text-xs text-stone-500 dark:text-stone-400">${scoreLabel} (${selectedModel})</p>
			</div>
		`;
		watchlistBox.appendChild(item);
	});
}
