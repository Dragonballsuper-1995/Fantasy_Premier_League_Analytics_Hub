// Configure Tailwind for dark mode via class
// This must run after the CDN script is loaded
try {
	if (typeof tailwind !== 'undefined') {
		tailwind.config = { darkMode: 'class' };
	}
} catch (_) {}
