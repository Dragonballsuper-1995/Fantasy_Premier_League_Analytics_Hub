// Common theme + animated background behavior
(function() {
	function applyTheme(theme) {
		var isDark = theme === 'dark';
		document.documentElement.classList.toggle('dark', isDark);
		try {
			var sun = document.getElementById('themeIconSun');
			var moon = document.getElementById('themeIconMoon');
			if (sun && moon) {
				sun.classList.toggle('hidden', isDark);
				moon.classList.toggle('hidden', !isDark);
			}
			var bg = document.getElementById('animated-bg');
			if (bg) {
				bg.style.background = isDark
					? 'radial-gradient(circle at 50% 50%, rgba(217, 119, 6, 0.1), rgba(28, 25, 23, 0) 40%)'
					: 'radial-gradient(circle at 50% 50%, rgba(217, 119, 6, 0.2), rgba(255, 251, 235, 0) 40%)';
			}
		} catch (_) {}
		// Notify pages to react (e.g., re-render charts)
		var evt;
		try { evt = new CustomEvent('themechange', { detail: { isDark: isDark } }); }
		catch (_) { evt = document.createEvent('Event'); evt.initEvent('themechange', true, true); }
		document.dispatchEvent(evt);
	}

	// Expose to window for optional direct use
	window.applyTheme = applyTheme;

	document.addEventListener('DOMContentLoaded', function() {
		var savedTheme = localStorage.getItem('theme') || 'light';
		applyTheme(savedTheme);

		var toggle = document.getElementById('themeToggle');
		if (toggle) {
			toggle.addEventListener('click', function() {
				var newTheme = document.documentElement.classList.contains('dark') ? 'light' : 'dark';
				try { localStorage.setItem('theme', newTheme); } catch (_) {}
				applyTheme(newTheme);
			});
		}

		var bg = document.getElementById('animated-bg');
		if (bg) {
			window.addEventListener('mousemove', function(e) {
				var x = Math.round((e.clientX / window.innerWidth) * 100);
				var y = Math.round((e.clientY / window.innerHeight) * 100);
				var isDark = document.documentElement.classList.contains('dark');
				bg.style.background = isDark
					? 'radial-gradient(circle at ' + x + '% ' + y + '%, rgba(217, 119, 6, 0.1), rgba(28, 25, 23, 0) 40%)'
					: 'radial-gradient(circle at ' + x + '% ' + y + '%, rgba(217, 119, 6, 0.2), rgba(255, 251, 235, 0) 40%)';
			});
		}
	});
})();
