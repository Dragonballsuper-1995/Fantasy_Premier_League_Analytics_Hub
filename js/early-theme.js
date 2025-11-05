// Apply saved theme ASAP to avoid FOUC
(function() {
	try {
		var savedTheme = localStorage.getItem('theme') || 'light';
		if (savedTheme === 'dark') {
			document.documentElement.classList.add('dark');
		} else {
			document.documentElement.classList.remove('dark');
		}
	} catch (_) {}
})();
