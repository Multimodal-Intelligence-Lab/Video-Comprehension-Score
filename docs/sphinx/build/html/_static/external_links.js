// Make external links open in new tabs
document.addEventListener('DOMContentLoaded', function() {
    // Find all external links
    var links = document.querySelectorAll('a[href^="http"], a[href^="https"]');
    
    for (var i = 0; i < links.length; i++) {
        var link = links[i];
        var href = link.getAttribute('href');
        
        // Check if it's an external link (not same domain)
        if (href && !href.includes(window.location.hostname)) {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
            
            // Add external link class for styling
            if (!link.classList.contains('external')) {
                link.classList.add('external');
            }
        }
    }
});