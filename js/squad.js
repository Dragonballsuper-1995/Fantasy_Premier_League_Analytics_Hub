// Squad builder logic extracted from inline script
let ALL_PLAYER_DATA = [], ALL_MODELS = [];

const modelSelector = document.getElementById('modelSelector');
const loadingStatus = document.getElementById('loadingStatus');
const formationSelector = document.getElementById('formationSelector');
const buildSquadButton = document.getElementById('buildSquadButton');
const squadResult = document.getElementById('squadResult');

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
	} catch (error) {
		console.error("Error loading predictions.json:", error);
		loadingStatus.innerHTML = `<strong class="text-red-600 dark:text-red-400">Error: Could not load data.</strong><br>${error.message}`;
	}
}
document.addEventListener('DOMContentLoaded', loadData);

buildSquadButton.addEventListener('click', buildOptimalSquad);

function buildOptimalSquad() {
	buildSquadButton.disabled = true;
	buildSquadButton.textContent = "Building...";
	squadResult.innerHTML = `<p class="text-stone-500 dark:text-stone-400 text-center">Calculating optimal squad...</p>`;

	setTimeout(() => {
		try {
			const selectedModel = modelSelector.value;
			const selectedFormation = formationSelector.value;
			const budget = 100.0;

			const playerPool = ALL_PLAYER_DATA.map(p => {
				const score = p.upcoming_predictions?.[0]?.predictions?.[selectedModel] ?? 0;
				const cost = p.cost || 0;
				const value = (cost > 0) ? score / cost : 0;
				return { ...p, score, value };
			}).filter(p => p.cost > 0 && p.score > 0);

			playerPool.sort((a, b) => b.value - a.value);

			const positionLimits = { GK: 2, DEF: 5, MID: 5, FWD: 3 };
			const squadSize = 15;
			let squad = [], positionCounts = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
			let teamCounts = {}, budgetRemaining = budget;

			for (const player of playerPool) {
				const team = player.team || 'Unknown';
				const position = player.position;
				if (!position) continue;
				const canAddPosition = positionCounts[position] < positionLimits[position];
				const canAddTeam = (teamCounts[team] || 0) < 3;
				const canAddBudget = budgetRemaining >= player.cost;
				if (canAddPosition && canAddTeam && canAddBudget) {
					squad.push(player);
					positionCounts[position]++;
					teamCounts[team] = (teamCounts[team] || 0) + 1;
					budgetRemaining -= player.cost;
				}
				if (squad.length === squadSize) break;
			}

			if (squad.length < squadSize) {
				throw new Error(`Greedy algorithm failed to build a full 15-man squad. Only found ${squad.length} players.`);
			}

			squad.sort((a,b) => b.score - a.score);
			const captainId = squad[0].id;
			const viceCaptainId = squad[1].id;
			const topPlayerIds = squad.slice(0, 3).map(p => p.id);

			const [def, mid, fwd] = selectedFormation.split('-').map(Number);
			const formationMap = { GK: 1, DEF: def, MID: mid, FWD: fwd };

			let startingXI = [], bench = [];
			const gks = squad.filter(p => p.position === 'GK').sort((a,b) => b.score - a.score);
			const defs = squad.filter(p => p.position === 'DEF').sort((a,b) => b.score - a.score);
			const mids = squad.filter(p => p.position === 'MID').sort((a,b) => b.score - a.score);
			const fwds = squad.filter(p => p.position === 'FWD').sort((a,b) => b.score - a.score);

			startingXI.push(...gks.splice(0, formationMap.GK));
			startingXI.push(...defs.splice(0, formationMap.DEF));
			startingXI.push(...mids.splice(0, formationMap.MID));
			startingXI.push(...fwds.splice(0, formationMap.FWD));
			bench.push(...gks, ...defs, ...mids, ...fwds);

			const benchGK = bench.filter(p => p.position === 'GK');
			const benchOutfield = bench.filter(p => p.position !== 'GK');
			bench = [...benchGK, ...benchOutfield];

			renderSquad(startingXI, bench, budget - budgetRemaining, selectedModel, captainId, viceCaptainId, topPlayerIds);

		} catch (error) {
			console.error("Squad build error:", error);
			squadResult.innerHTML = `<p class="text-red-600 dark:text-red-400 text-center font-semibold">${error.message}</p>`;
		} finally {
			buildSquadButton.disabled = false;
			buildSquadButton.textContent = "Build Optimal Squad (£100m)";
		}
	}, 100);
}

const createPlayerBox = (player, captainId, viceCaptainId, topPlayerIds) => {
	let badgeHtml = '<div class="player-badges-container">';
	if (topPlayerIds.includes(player.id)) badgeHtml += '<span class="player-badge player-star">★</span>';
	if (player.id === captainId) badgeHtml += '<span class="player-badge player-captain">C</span>';
	else if (player.id === viceCaptainId) badgeHtml += '<span class="player-badge player-vice-captain">V</span>';
	badgeHtml += '</div>';
	return `
		<div class="player-box">
			${badgeHtml}
			<span class="player-box-name">${player.web_name}</span>
			<span class="player-box-team">${player.team}</span>
			<div class="player-box-info">
				<span class="player-box-score">${player.score.toFixed(1)} xPts</span>
				<span class="player-box-price">£${player.cost.toFixed(1)}m</span>
			</div>
		</div>`;
};

const createBenchPlayerBox = (player, captainId, viceCaptainId, topPlayerIds) => {
	let badgeHtml = '<div class="player-badges-container">';
	if (topPlayerIds.includes(player.id)) badgeHtml += '<span class="player-badge player-star">★</span>';
	if (player.id === captainId) badgeHtml += '<span class="player-badge player-captain">C</span>';
	else if (player.id === viceCaptainId) badgeHtml += '<span class="player-badge player-vice-captain">V</span>';
	badgeHtml += '</div>';
	return `
		<div class="bench-player-box">
			${badgeHtml}
			<span class="player-box-name">${player.web_name}</span>
			<span class="player-box-team">${player.team}</span>
			<div class="player-box-info">
				<span class="player-box-score">${player.score.toFixed(1)} xPts</span>
				<span class="player-box-price">£${player.cost.toFixed(1)}m</span>
			</div>
		</div>`;
};

const createPlayerListRow = (player, captainId, viceCaptainId, topPlayerIds) => {
	let badgeHtml = '<div class="squad-list-badges">';
	if (topPlayerIds.includes(player.id)) badgeHtml += '<span class="player-badge player-star">★</span>';
	if (player.id === captainId) badgeHtml += '<span class="player-badge player-captain">C</span>';
	else if (player.id === viceCaptainId) badgeHtml += '<span class="player-badge player-vice-captain">V</span>';
	badgeHtml += '</div>';
	return `
		<div class="squad-list-player">
			${badgeHtml}
			<div class="squad-list-info">
				<span class="squad-list-name">${player.web_name}</span>
				<span class="squad-list-team">(${player.position} / ${player.team})</span>
			</div>
			<div class="squad-list-stats">
				<span class="squad-list-score">${player.score.toFixed(1)} xPts</span>
				<div class="squad-list-price">£${player.cost.toFixed(1)}m</div>
			</div>
		</div>`;
};

function renderSquadPitch(startingXI, captainId, viceCaptainId, topPlayerIds) {
	const gks = startingXI.filter(p => p.position === 'GK');
	const defs = startingXI.filter(p => p.position === 'DEF');
	const mids = startingXI.filter(p => p.position === 'MID');
	const fwds = startingXI.filter(p => p.position === 'FWD');
	const renderRow = (players) => players.map(p => createPlayerBox(p, captainId, viceCaptainId, topPlayerIds)).join('');
	let pitchHtml = '<div class="squad-pitch">';
	pitchHtml += `<div class="pitch-line-half"></div><div class="pitch-line-box-top"></div><div class="pitch-line-box-bottom"></div>`;
	pitchHtml += '<div class="pitch-rows-overlay">';
	if (gks.length > 0) pitchHtml += `<div class="pitch-row gk-row">${renderRow(gks)}</div>`;
	if (defs.length > 0) pitchHtml += `<div class="pitch-row def-row">${renderRow(defs)}</div>`;
	if (mids.length > 0) pitchHtml += `<div class="pitch-row mid-row">${renderRow(mids)}</div>`;
	if (fwds.length > 0) pitchHtml += `<div class="pitch-row fwd-row">${renderRow(fwds)}</div>`;
	pitchHtml += '</div></div>';
	return pitchHtml;
}

function renderStartingXIList(startingXI, captainId, viceCaptainId, topPlayerIds) {
	let listHtml = '<div class="squad-list-container">';
	const gks = startingXI.filter(p => p.position === 'GK');
	const defs = startingXI.filter(p => p.position === 'DEF');
	const mids = startingXI.filter(p => p.position === 'MID');
	const fwds = startingXI.filter(p => p.position === 'FWD');
	const renderListRow = (players) => players.map(p => createPlayerListRow(p, captainId, viceCaptainId, topPlayerIds)).join('');
	listHtml += renderListRow(gks);
	listHtml += renderListRow(defs);
	listHtml += renderListRow(mids);
	listHtml += renderListRow(fwds);
	listHtml += '</div>';
	return listHtml;
}

const renderBenchPlayers = (benchPlayers, captainId, viceCaptainId, topPlayerIds) => {
	const benchGKs = benchPlayers.filter(p => p.position === 'GK');
	const benchDEFs = benchPlayers.filter(p => p.position === 'DEF');
	const benchMIDs = benchPlayers.filter(p => p.position === 'MID');
	const benchFWDs = benchPlayers.filter(p => p.position === 'FWD');
	const createGroupHtml = (label, players) => {
		if (players.length === 0) return '';
		return `
			<div class="bench-position-group">
				<h4 class="bench-position-label">${label}</h4>
				<div class="bench-players-row">
					${players.map(p => createBenchPlayerBox(p, captainId, viceCaptainId, topPlayerIds)).join('')}
				</div>
			</div>`;
	};
	let benchHtml = `<h3 class="text-lg font-semibold mb-2 mt-4 text-center">Bench</h3>`;
	benchHtml += '<div class="squad-bench-container">';
	if (benchPlayers.length === 0) benchHtml += '<p class="text-stone-500 dark:text-stone-400 text-center text-sm">No players on the bench.</p>';
	else {
		benchHtml += createGroupHtml('Goalkeeper', benchGKs);
		benchHtml += createGroupHtml('Defenders', benchDEFs);
		benchHtml += createGroupHtml('Midfielders', benchMIDs);
		benchHtml += createGroupHtml('Forwards', benchFWDs);
	}
	benchHtml += '</div>';
	return benchHtml;
};

function renderSquad(startingXI, bench, totalCost, model, captainId, viceCaptainId, topPlayerIds) {
	const totalPoints = startingXI.reduce((sum, p) => sum + p.score, 0);
	squadResult.innerHTML = `
		<div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
			<div class="bg-amber-100 dark:bg-stone-700 p-4 rounded-lg text-center">
				<p class="text-sm font-medium text-amber-700 dark:text-amber-300 uppercase">Model</p>
				<p class="text-2xl font-bold text-amber-700 dark:text-amber-200">${model}</p>
			</div>
			<div class="bg-stone-100 dark:bg-stone-700 p-4 rounded-lg text-center">
				<p class="text-sm font-medium text-stone-500 dark:text-stone-400 uppercase">Total Cost</p>
				<p class="text-2xl font-bold">£${totalCost.toFixed(1)}m / £100.0m</p>
			</div>
			<div class="bg-stone-100 dark:bg-stone-700 p-4 rounded-lg text-center">
				<p class="text-sm font-medium text-stone-500 dark:text-stone-400 uppercase">Starting XI xPts</p>
				<p class="text-2xl font-bold text-green-600 dark:text-green-500">${totalPoints.toFixed(1)}</p>
			</div>
		</div>

		<div class="flex flex-col items-center mb-2">
			<div class="flex justify-center items-center gap-2">
				<h3 class="text-lg font-semibold text-center">Starting XI</h3>
				<div class="flex items-center bg-stone-200 dark:bg-stone-700 rounded-lg p-0.5">
					<button id="squadViewTogglePitch" class="px-3 py-1 text-sm font-bold rounded-md bg-white dark:bg-stone-900 shadow">Pitch</button>
					<button id="squadViewToggleList" class="px-3 py-1 text-sm font-medium rounded-md text-stone-600 dark:text-stone-300">List</button>
				</div>
			</div>
		</div>

		<div id="squadPitchView">${renderSquadPitch(startingXI, captainId, viceCaptainId, topPlayerIds)}</div>
		<div id="squadListView" class="hidden">${renderStartingXIList(startingXI, captainId, viceCaptainId, topPlayerIds)}</div>

		${renderBenchPlayers(bench, captainId, viceCaptainId, topPlayerIds)}
	`;
	document.getElementById('squadViewTogglePitch').addEventListener('click', toggleSquadView);
	document.getElementById('squadViewToggleList').addEventListener('click', toggleSquadView);
}

function toggleSquadView(e) {
	const isPitch = e.currentTarget.id === 'squadViewTogglePitch';
	const pitchButton = document.getElementById('squadViewTogglePitch');
	const listButton = document.getElementById('squadViewToggleList');
	document.getElementById('squadPitchView').classList.toggle('hidden', !isPitch);
	document.getElementById('squadListView').classList.toggle('hidden', isPitch);
	pitchButton.classList.toggle('bg-white', isPitch);
	pitchButton.classList.toggle('dark:bg-stone-900', isPitch);
	pitchButton.classList.toggle('shadow', isPitch);
	pitchButton.classList.toggle('font-bold', isPitch);
	pitchButton.classList.toggle('text-stone-600', !isPitch);
	pitchButton.classList.toggle('dark:text-stone-300', !isPitch);
	pitchButton.classList.toggle('font-medium', !isPitch);
	listButton.classList.toggle('bg-white', !isPitch);
	listButton.classList.toggle('dark:bg-stone-900', !isPitch);
	listButton.classList.toggle('shadow', !isPitch);
	listButton.classList.toggle('font-bold', !isPitch);
	listButton.classList.toggle('text-stone-600', isPitch);
	listButton.classList.toggle('dark:text-stone-300', isPitch);
	listButton.classList.toggle('font-medium', isPitch);
}
