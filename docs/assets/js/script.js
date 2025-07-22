// --- DEBUG FUNCTION - Call from browser console ---
window.debugDemoLoading = function(componentId = 'introduction') {
    console.log('🔧 Manual debug test for:', componentId);
    
    const componentCards = mainContentContainer.querySelectorAll('.component-card');
    const modal = mainContentContainer.querySelector('#componentModal');
    const iframe = mainContentContainer.querySelector('#componentIframe');
    
    console.log('🔍 Debug results:', {
        mainContentContainer: !!mainContentContainer,
        componentCards: componentCards.length,
        modal: !!modal,
        iframe: !!iframe,
        modalInDocument: !!document.getElementById('componentModal'),
        iframeInDocument: !!document.getElementById('componentIframe')
    });
    
    if (componentCards.length > 0) {
        const targetCard = Array.from(componentCards).find(card => 
            card.getAttribute('data-component') === componentId
        );
        if (targetCard) {
            console.log('🎯 Found target card, simulating click');
            targetCard.click();
        } else {
            console.log('❌ Target card not found');
            console.log('Available components:', Array.from(componentCards).map(card => card.getAttribute('data-component')));
        }
    }
};

document.addEventListener('DOMContentLoaded', function () {
    // --- Main Elements ---
    const primaryTabButtons = document.querySelectorAll('#primaryTabsNav .tab-button');
    const mainContentContainer = document.getElementById('mainContentContainer');
    const stickyHeader = document.querySelector('.sticky-header');
    const scrollToTopBtn = document.getElementById('scrollToTopBtn');

    // --- State Variables ---
    let currentStickyHeaderHeight = stickyHeader ? stickyHeader.offsetHeight : 0;
    let apiNavLinks = [], apiRightPane, apiContentSections = [];
    let contentSectionObserver, apiSectionObserver;

    // --- Helper: Check if element is in viewport (still useful for general fade-ins) ---
    function isElementInViewport(el) {
        if (!el) return false;
        const rect = el.getBoundingClientRect();
        return (
            rect.top < window.innerHeight && rect.bottom >= 0 &&
            rect.left < window.innerWidth && rect.right >= 0
        );
    }

    // --- Network Quality Detection ---
    function detectNetworkQuality() {
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
        
        if (connection) {
            const { effectiveType, downlink, rtt } = connection;
            
            // Determine loading strategy based on network quality
            if (effectiveType === 'slow-2g' || effectiveType === '2g' || downlink < 0.5 || rtt > 2000) {
                return 'slow';
            } else if (effectiveType === '3g' || downlink < 1.5) {
                return 'medium';
            } else {
                return 'fast';
            }
        }
        
        // Fallback: measure actual loading time
        return 'unknown';
    }

    // --- Progressive Enhancement Setup ---
    document.documentElement.classList.add('js');
    
    // Apply network quality class to body
    const networkQuality = detectNetworkQuality();
    document.body.classList.add(`${networkQuality}-network`);
    
    // Initialize cursor tracking for dynamic modal backgrounds
    initializeCursorTracking();

    // --- Page-Specific Skeleton Templates ---
    function createSkeletonLoader(tabId) {
        const skeletons = {
            playground: `
                <div id="playgroundContent" class="tab-inner-content">
                    <article class="bg-white rounded-xl shadow-xl overflow-hidden">
                        <!-- Hero Section Skeleton -->
                        <section class="py-10 px-4 sm:px-6 lg:px-8" style="background: linear-gradient(45deg, #0d9488, #0f766e);">
                            <div class="container mx-auto">
                                <div class="flex flex-col md:flex-row md:items-stretch gap-8 lg:gap-12">
                                    <!-- Video Section Skeleton -->
                                    <div class="md:w-1/2 lg:w-3/5 w-full flex md:items-start">
                                        <div class="skeleton-video rounded-lg shadow-2xl border-2 border-teal-500/50" style="aspect-ratio: 16/9; background: linear-gradient(90deg, #1f2937 25%, #374151 50%, #1f2937 75%); background-size: 200px 100%; animation: shimmer 1.5s infinite linear;"></div>
                                    </div>
                                    
                                    <!-- Title and Links Section Skeleton -->
                                    <div class="md:w-1/2 lg:w-2/5 w-full text-left flex flex-col">
                                        <div class="flex-grow">
                                            <div class="skeleton-title skeleton mb-3" style="background: linear-gradient(90deg, rgba(255,255,255,0.3) 25%, rgba(255,255,255,0.5) 50%, rgba(255,255,255,0.3) 75%); height: 3rem; width: 90%;"></div>
                                            <div class="skeleton-text skeleton mb-6" style="background: linear-gradient(90deg, rgba(255,255,255,0.2) 25%, rgba(255,255,255,0.4) 50%, rgba(255,255,255,0.2) 75%); height: 1.5rem; width: 85%;"></div>
                                            
                                            <!-- 6 Link Boxes Skeleton -->
                                            <div class="grid grid-cols-2 gap-3 mb-6">
                                                <div class="skeleton-hero-link"><div class="skeleton-icon mb-2" style="width: 32px; height: 32px;"></div><div class="skeleton-text short" style="background: rgba(255,255,255,0.3); height: 0.875rem; width: 70%;"></div></div>
                                                <div class="skeleton-hero-link"><div class="skeleton-icon mb-2" style="width: 32px; height: 32px;"></div><div class="skeleton-text short" style="background: rgba(255,255,255,0.3); height: 0.875rem; width: 70%;"></div></div>
                                                <div class="skeleton-hero-link"><div class="skeleton-icon mb-2" style="width: 32px; height: 32px;"></div><div class="skeleton-text short" style="background: rgba(255,255,255,0.3); height: 0.875rem; width: 70%;"></div></div>
                                                <div class="skeleton-hero-link"><div class="skeleton-icon mb-2" style="width: 32px; height: 32px;"></div><div class="skeleton-text short" style="background: rgba(255,255,255,0.3); height: 0.875rem; width: 70%;"></div></div>
                                                <div class="skeleton-hero-link"><div class="skeleton-icon mb-2" style="width: 32px; height: 32px;"></div><div class="skeleton-text short" style="background: rgba(255,255,255,0.3); height: 0.875rem; width: 70%;"></div></div>
                                                <div class="skeleton-hero-link"><div class="skeleton-icon mb-2" style="width: 32px; height: 32px;"></div><div class="skeleton-text short" style="background: rgba(255,255,255,0.3); height: 0.875rem; width: 70%;"></div></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </section>

                        <!-- Component Discovery Section Skeleton -->
                        <div class="component-discovery-section">
                            <div class="discovery-header">
                                <div class="skeleton-title skeleton discovery-title mb-4" style="width: 60%; margin: 0 auto;"></div>
                                <div class="skeleton-text skeleton discovery-subtitle" style="width: 70%; margin: 0 auto;"></div>
                            </div>
                            
                            <div class="components-grid">
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                                <div class="skeleton-card component-card" style="min-height: 280px;"><div class="card-header"><div class="skeleton-icon"></div><div class="skeleton" style="width: 80px; height: 24px; border-radius: 12px;"></div></div><div class="card-content"><div class="skeleton-title skeleton card-title"></div><div class="skeleton-text skeleton long mb-2"></div><div class="skeleton-text skeleton medium mb-4"></div><div class="flex gap-2"><div class="skeleton" style="width: 60px; height: 20px; border-radius: 10px;"></div><div class="skeleton" style="width: 70px; height: 20px; border-radius: 10px;"></div></div></div><div class="card-footer"><div class="skeleton-button skeleton"></div></div></div>
                            </div>
                        </div>
                    </article>
                </div>
            `,
            
            example: `
                <div id="exampleContent" class="tab-inner-content">
                    <article class="bg-white rounded-xl shadow-xl overflow-hidden">
                        <div class="p-6 sm:p-8 lg:p-10">
                            <section class="content-section">
                                <!-- Main Title Skeleton -->
                                <div class="skeleton-title skeleton text-center mb-4" style="width: 80%; margin: 0 auto; height: 3rem;"></div>
                                <div class="skeleton-text skeleton text-center mb-10" style="width: 90%; margin: 0 auto; height: 1.5rem;"></div>

                                <div class="space-y-6">
                                    <!-- Step 1 Skeleton -->
                                    <div class="text-center">
                                        <div class="flex items-center justify-center gap-3 mb-6">
                                            <div class="skeleton-icon bg-teal-100"></div>
                                            <div class="skeleton-title skeleton" style="width: 400px; height: 2rem;"></div>
                                        </div>
                                    </div>
                                    
                                    <!-- Two Column Text Boxes Skeleton -->
                                    <div class="flex flex-col md:flex-row justify-around items-stretch md:space-x-6 space-y-6 md:space-y-0">
                                        <div class="bg-white/90 backdrop-blur-sm border border-teal-200 rounded-xl p-6 shadow-lg md:w-1/2">
                                            <div class="flex items-center gap-2 mb-4">
                                                <div class="skeleton" style="width: 8px; height: 8px; border-radius: 50%; background: #0d9488;"></div>
                                                <div class="skeleton-text skeleton" style="width: 200px; height: 1.5rem;"></div>
                                            </div>
                                            <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 min-h-32">
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                            </div>
                                        </div>
                                        <div class="bg-white/90 backdrop-blur-sm border border-teal-200 rounded-xl p-6 shadow-lg md:w-1/2">
                                            <div class="flex items-center gap-2 mb-4">
                                                <div class="skeleton" style="width: 8px; height: 8px; border-radius: 50%; background: #0d9488;"></div>
                                                <div class="skeleton-text skeleton" style="width: 200px; height: 1.5rem;"></div>
                                            </div>
                                            <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 min-h-32">
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                                <div class="skeleton-text skeleton mb-2"></div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Additional Steps Skeleton -->
                                    <div class="bg-white/90 backdrop-blur-sm border border-teal-200 rounded-xl p-6 shadow-lg mb-6">
                                        <div class="flex items-center gap-3 mb-4">
                                            <div class="skeleton-icon bg-teal-100"></div>
                                            <div class="skeleton-title skeleton" style="width: 300px; height: 2rem;"></div>
                                        </div>
                                        <div class="skeleton-text skeleton long mb-2"></div>
                                        <div class="skeleton-text skeleton medium mb-2"></div>
                                        <div class="skeleton-text skeleton short"></div>
                                    </div>
                                    <div class="bg-white/90 backdrop-blur-sm border border-teal-200 rounded-xl p-6 shadow-lg mb-6">
                                        <div class="flex items-center gap-3 mb-4">
                                            <div class="skeleton-icon bg-teal-100"></div>
                                            <div class="skeleton-title skeleton" style="width: 300px; height: 2rem;"></div>
                                        </div>
                                        <div class="skeleton-text skeleton long mb-2"></div>
                                        <div class="skeleton-text skeleton medium mb-2"></div>
                                        <div class="skeleton-text skeleton short"></div>
                                    </div>
                                    <div class="bg-white/90 backdrop-blur-sm border border-teal-200 rounded-xl p-6 shadow-lg mb-6">
                                        <div class="flex items-center gap-3 mb-4">
                                            <div class="skeleton-icon bg-teal-100"></div>
                                            <div class="skeleton-title skeleton" style="width: 300px; height: 2rem;"></div>
                                        </div>
                                        <div class="skeleton-text skeleton long mb-2"></div>
                                        <div class="skeleton-text skeleton medium mb-2"></div>
                                        <div class="skeleton-text skeleton short"></div>
                                    </div>
                                    <div class="bg-white/90 backdrop-blur-sm border border-teal-200 rounded-xl p-6 shadow-lg mb-6">
                                        <div class="flex items-center gap-3 mb-4">
                                            <div class="skeleton-icon bg-teal-100"></div>
                                            <div class="skeleton-title skeleton" style="width: 300px; height: 2rem;"></div>
                                        </div>
                                        <div class="skeleton-text skeleton long mb-2"></div>
                                        <div class="skeleton-text skeleton medium mb-2"></div>
                                        <div class="skeleton-text skeleton short"></div>
                                    </div>
                                </div>
                            </section>
                        </div>
                    </article>
                </div>
            `,

            api: `
                <div id="apiContent" class="tab-inner-content">
                    <article class="article-card">
                        <div class="p-6 sm:p-8 lg:p-10">
                            <section class="content-section">
                                <!-- API Title Skeleton -->
                                <div class="skeleton-title skeleton mb-4" style="width: 400px; height: 2.5rem;"></div>
                                <div class="skeleton-text skeleton mb-6" style="width: 90%; height: 1.25rem;"></div>

                                <!-- Iframe Container Skeleton -->
                                <div class="w-full bg-white/95 backdrop-blur-sm border border-teal-200/50 rounded-xl shadow-2xl overflow-hidden relative" style="height: 85vh; min-height: 700px;">
                                    <div class="absolute inset-0 bg-gradient-to-br from-teal-50/95 to-white/95 backdrop-blur-sm flex items-center justify-center">
                                        <div class="text-center p-8">
                                            <div class="relative mb-6">
                                                <div class="loading-spinner"></div>
                                            </div>
                                            <div class="skeleton-title skeleton mb-2" style="width: 250px; height: 1.5rem; margin: 0 auto;"></div>
                                            <div class="skeleton-text skeleton" style="width: 300px; height: 1rem; margin: 0 auto;"></div>
                                        </div>
                                    </div>
                                </div>
                            </section>
                        </div>
                    </article>
                </div>
            `
        };
        
        return skeletons[tabId] || skeletons.playground;
    }

    // --- Show Loading State ---
    function showLoadingState(targetTabId = 'playground') {
        const loadingHTML = `
            <div class="loading-container" id="loadingContainer">
                ${createSkeletonLoader(targetTabId)}
            </div>
        `;
        mainContentContainer.innerHTML = loadingHTML;
    }

    // --- Staggered Content Loading ---
    function applyStaggeredLoading() {
        // Apply stagger classes to various content elements
        const staggerElements = [
            '.component-card',
            '.content-section', 
            '.info-card',
            '.hero-link-box',
            '.flowchart-block',
            '.enhanced-placeholder'
        ];

        staggerElements.forEach(selector => {
            const elements = mainContentContainer.querySelectorAll(selector);
            elements.forEach((el, index) => {
                el.classList.add('stagger-item');
                
                // Stagger the loading animation
                setTimeout(() => {
                    el.classList.add('loaded');
                }, (index + 1) * 100);
            });
        });

        // Apply content-loaded class to main sections
        const mainSections = mainContentContainer.querySelectorAll('#hero-section, .component-discovery-section, .getting-started-section');
        mainSections.forEach((section, index) => {
            section.classList.add('content-loaded');
            setTimeout(() => {
                section.classList.add('visible');
            }, (index + 1) * 200);
        });
    }

    // --- Adaptive Loading Strategy ---
    function getLoadingStrategy() {
        const networkQuality = detectNetworkQuality();
        const strategies = {
            slow: {
                skeletonDuration: 800,
                staggerDelay: 200,
                enableLazyLoading: true,
                reduceAnimations: true
            },
            medium: {
                skeletonDuration: 500,
                staggerDelay: 150,
                enableLazyLoading: true,
                reduceAnimations: false
            },
            fast: {
                skeletonDuration: 300,
                staggerDelay: 100,
                enableLazyLoading: false,
                reduceAnimations: false
            },
            unknown: {
                skeletonDuration: 500,
                staggerDelay: 150,
                enableLazyLoading: true,
                reduceAnimations: false
            }
        };
        
        return strategies[networkQuality];
    }

    // --- Error State Handler ---
    function showErrorState(error, retryCallback) {
        const errorHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-triangle text-2xl mb-2"></i>
                <h3 class="text-lg font-semibold mb-2">Failed to Load Content</h3>
                <p class="mb-4">${error.message || 'An unexpected error occurred while loading the content.'}</p>
                <button class="retry-button" onclick="(${retryCallback.toString()})()">
                    <i class="fas fa-redo mr-2"></i>
                    Try Again
                </button>
            </div>
        `;
        mainContentContainer.innerHTML = errorHTML;
    }

    // --- Update Sticky Header Height on Resize ---
    window.addEventListener('resize', () => {
        currentStickyHeaderHeight = stickyHeader ? stickyHeader.offsetHeight : 0;
        if (contentSectionObserver) setupContentSectionObserver();
        if (apiSectionObserver) setupApiTabObserver();
    });

    // --- Scroll to Element Function (generalized) ---
    function scrollToElement(targetElementId) {
        const element = document.getElementById(targetElementId);
        if (element) {
            const offset = currentStickyHeaderHeight + 20; 
            const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
            const offsetPosition = elementPosition - offset;
            
            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
    }

    // --- Smart Modal System ---
    const modalSystem = {
        preloadCache: new Map(),
        visitedComponents: new Set(),
        modalMemory: new Map(),
        hoverTimeouts: new Map(),
        preloadTimeouts: new Map()
    };

    // --- Component Discovery Management ---
    function initializeComponentDiscovery() {
        console.log('🚀 initializeComponentDiscovery called');
        console.log('📦 mainContentContainer:', mainContentContainer);
        console.log('📄 mainContentContainer.innerHTML length:', mainContentContainer.innerHTML.length);
        
        // NOTE: These elements are inside the dynamically loaded 'playground.html'
        const componentCards = mainContentContainer.querySelectorAll('.component-card');
        const modal = mainContentContainer.querySelector('#componentModal');
        const iframe = mainContentContainer.querySelector('#componentIframe');
        const modalTitle = mainContentContainer.querySelector('#modalTitle');
        const modalSubtitle = mainContentContainer.querySelector('#modalSubtitle');
        const modalIcon = mainContentContainer.querySelector('.modal-component-icon i');
        const closeBtn = mainContentContainer.querySelector('#closeComponentModal');
        const prevBtn = mainContentContainer.querySelector('#prevComponent');
        const nextBtn = mainContentContainer.querySelector('#nextComponent');
        const loadingDiv = mainContentContainer.querySelector('#iframeLoading');
        const comingSoonDiv = mainContentContainer.querySelector('#iframeComingSoon');
        
        console.log('🔍 Element search results:', {
            componentCards: componentCards.length,
            modal: !!modal,
            iframe: !!iframe,
            modalTitle: !!modalTitle,
            modalSubtitle: !!modalSubtitle,
            modalIcon: !!modalIcon,
            closeBtn: !!closeBtn,
            prevBtn: !!prevBtn,
            nextBtn: !!nextBtn,
            loadingDiv: !!loadingDiv,
            comingSoonDiv: !!comingSoonDiv
        });

        // Restore visited states from localStorage
        restoreModalMemory(componentCards);
        
        let currentComponentIndex = 0;
        // CORRECTED: This array now matches the components available in playground.html
        const components = [
            {
                id: 'introduction',
                title: 'Introduction to VCS',
                subtitle: 'Learn about video description evaluation challenges',
                icon: 'fas fa-play-circle',
                iframe: 'widgets/introduction/index.html',
                available: true
            },
            {
                id: 'gas',
                title: 'Global Alignment Score (GAS)',
                subtitle: 'Overall thematic similarity measurement',
                icon: 'fas fa-globe-americas',
                iframe: 'widgets/gas/index.html',
                available: true
            },
            {
                id: 'mapping-windows',
                title: 'Mapping Windows',
                subtitle: 'Define alignment ranges for different text lengths',
                icon: 'fas fa-window-restore',
                iframe: 'widgets/mapping-window/index.html',
                available: true
            },
            {
                id: 'best-matching',
                title: 'Best Matching Algorithm',
                subtitle: 'Establish robust chunk correspondences',
                icon: 'fas fa-magic',
                iframe: 'widgets/best-match/index.html',
                available: true
            },
            {
                id: 'las',
                title: 'Local Alignment Score (LAS)',
                subtitle: 'Fine-grained semantic quality assessment',
                icon: 'fas fa-tasks',
                iframe: 'widgets/las/index.html',
                available: true
            },
            {
                id: 'nas-distance',
                title: 'Distance-based Narrative Alignment Score (NAS-D)',
                subtitle: 'Penalize position deviations within windows',
                icon: 'fas fa-route',
                iframe: 'widgets/distance-nas/index.html',
                available: true
            },
            {
                id: 'nas-line',
                title: 'Line-based Narrative Alignment Score (NAS-L)',
                subtitle: 'Evaluate chronological flow of elements',
                icon: 'fas fa-wave-square',
                iframe: 'widgets/line-nas/index.html',
                available: true
            },
            {
                id: 'lct-nas-d',
                title: 'Local Chronology Tolerance (LCT) Effect on NAS-D',
                subtitle: 'Configure chronology tolerance for Distance-based Narrative Alignment Score',
                icon: 'fas fa-sliders-h',
                iframe: 'widgets/lct-nas-d/index.html',
                available: true
            },
            {
                id: 'lct-nas-l',
                title: 'Local Chronology Tolerance (LCT) Effect on NAS-L',
                subtitle: 'Configure chronology tolerance for Line-based Narrative Alignment Score',
                icon: 'fas fa-sliders-h',
                iframe: 'widgets/lct-nas-l/index.html',
                available: true
            },
            {
                id: 'window-regularizer',
                title: 'Window Regularizer for Narrative Alignment',
                subtitle: 'Adjust NAS for extreme length disparities',
                icon: 'fas fa-window-maximize',
                iframe: 'widgets/window-regularizer/index.html',
                available: true
            },
            {
                id: 'nas',
                title: 'Narrative Alignment Score (NAS)',
                subtitle: 'Complete Narrative Alignment Score calculation',
                icon: 'fas fa-microchip',
                iframe: 'widgets/nas/index.html',
                available: true
            },
            {
                id: 'final-vcs',
                title: 'Final VCS Score',
                subtitle: 'Complete Video Comprehension Score calculation',
                icon: 'fas fa-calculator',
                iframe: 'widgets/vcs/index.html',
                available: true
            }
        ];
        
        if (!componentCards.length || !modal) {
            console.warn('Component discovery initialization failed:', {
                componentCards: componentCards.length,
                modal: !!modal,
                iframe: !!iframe,
                modalTitle: !!modalTitle,
                closeBtn: !!closeBtn
            });
            return;
        }
        
        // Mark modal as initialized to prevent double initialization
        modal.hasEventListeners = true;
        
        // Add enhanced interaction handlers to component cards
        componentCards.forEach((card) => {
            const demoButton = card.querySelector('.demo-button');
            if (demoButton && !demoButton.disabled) {
                const componentId = card.getAttribute('data-component');
                
                // Enhanced hover preloading
                card.addEventListener('mouseenter', () => handleCardHoverStart(card, componentId));
                card.addEventListener('mouseleave', () => handleCardHoverEnd(card, componentId));
                
                // Enhanced click with morphing
                card.addEventListener('click', () => {
                    console.log('🎯 Card clicked:', componentId);
                    const componentIndex = components.findIndex(c => c.id === componentId);
                    console.log('📍 Component index found:', componentIndex);
                    if (componentIndex > -1) {
                        console.log('✅ Opening modal for:', components[componentIndex]);
                        openComponentModalWithMorphing(card, componentIndex);
                    } else {
                        console.error('❌ Component not found:', componentId);
                    }
                });
            }
        });
        
        // --- Modal Memory Functions ---
        function restoreModalMemory(cards) {
            try {
                const visitedString = localStorage.getItem('vcs-visited-components');
                const memoryString = localStorage.getItem('vcs-modal-memory');
                
                if (visitedString) {
                    const visited = JSON.parse(visitedString);
                    visited.forEach(id => {
                        modalSystem.visitedComponents.add(id);
                        const card = mainContentContainer.querySelector(`[data-component="${id}"]`);
                        if (card) card.classList.add('visited');
                    });
                }
                
                if (memoryString) {
                    modalSystem.modalMemory = new Map(JSON.parse(memoryString));
                }
            } catch (error) {
                console.warn('Failed to restore modal memory:', error);
            }
        }

        function saveModalMemory() {
            try {
                localStorage.setItem('vcs-visited-components', 
                    JSON.stringify([...modalSystem.visitedComponents]));
                localStorage.setItem('vcs-modal-memory', 
                    JSON.stringify([...modalSystem.modalMemory]));
            } catch (error) {
                console.warn('Failed to save modal memory:', error);
            }
        }

        // --- Intelligent Preloading Functions ---
        function handleCardHoverStart(card, componentId) {
            const strategy = getLoadingStrategy();
            if (!strategy.enableLazyLoading) {
                // On fast networks, start preloading immediately
                startPreloading(card, componentId, 300);
            } else {
                // On slower networks, wait for meaningful hover
                modalSystem.hoverTimeouts.set(componentId, setTimeout(() => {
                    startPreloading(card, componentId, 800);
                }, 500));
            }
        }

        function handleCardHoverEnd(card, componentId) {
            const timeout = modalSystem.hoverTimeouts.get(componentId);
            if (timeout) {
                clearTimeout(timeout);
                modalSystem.hoverTimeouts.delete(componentId);
            }
            
            // Cancel preloading if it's in progress
            const preloadTimeout = modalSystem.preloadTimeouts.get(componentId);
            if (preloadTimeout) {
                clearTimeout(preloadTimeout);
                modalSystem.preloadTimeouts.delete(componentId);
                card.classList.remove('preloading');
            }
        }

        function startPreloading(card, componentId, duration) {
            if (modalSystem.preloadCache.has(componentId)) {
                card.classList.add('preloaded');
                return; // Already preloaded
            }

            card.classList.add('preloading');
            
            const component = components.find(c => c.id === componentId);
            if (!component || !component.iframe) return;

            // Create invisible iframe for preloading
            const preloadIframe = document.createElement('iframe');
            preloadIframe.style.position = 'absolute';
            preloadIframe.style.top = '-9999px';
            preloadIframe.style.left = '-9999px';
            preloadIframe.style.width = '1px';
            preloadIframe.style.height = '1px';
            preloadIframe.style.visibility = 'hidden';
            
            preloadIframe.onload = () => {
                modalSystem.preloadCache.set(componentId, {
                    loaded: Date.now(),
                    iframe: preloadIframe
                });
                card.classList.remove('preloading');
                card.classList.add('preloaded');
                console.log(`Preloaded component: ${componentId}`);
            };

            preloadIframe.onerror = () => {
                card.classList.remove('preloading');
                console.warn(`Failed to preload component: ${componentId}`);
            };

            document.body.appendChild(preloadIframe);
            
            modalSystem.preloadTimeouts.set(componentId, setTimeout(() => {
                preloadIframe.src = component.iframe;
                modalSystem.preloadTimeouts.delete(componentId);
            }, duration));
        }

        // --- Enhanced Modal Opening with Progressive Content Morphing ---
        function openComponentModalWithMorphing(sourceCard, index) {
            if (index < 0 || index >= components.length) return;

            const component = components[index];
            currentComponentIndex = index;

            // Record visit
            modalSystem.visitedComponents.add(component.id);
            sourceCard.classList.add('visited');
            saveModalMemory();

            // Start Progressive Content Morphing
            startContentMorphing(sourceCard, component);

            updateNavButtons();
        }

        // --- Progressive Content Morphing Implementation ---
        function startContentMorphing(sourceCard, component) {
            // Mark card as morphing
            sourceCard.classList.add('content-morphing');

            // Extract content elements from card
            const contentElements = extractContentElements(sourceCard);
            
            // Create visual bridge effects
            createVisualBridge(sourceCard);

            // Show modal overlay
            modal.classList.add('show', 'component-modal');
            document.body.style.overflow = 'hidden';

            // Update modal content immediately
            updateModalContent(component);

            // Hide modal header elements during morphing
            hideModalHeaderElements();

            // Get target positions in modal (after modal is visible)
            requestAnimationFrame(() => {
                const targetPositions = getModalTargetPositions();

                // Start content morphing sequence
                morphContentElements(contentElements, targetPositions, () => {
                    // Create and grow modal frame around settled content
                    createModalFrame(targetPositions, () => {
                        // Complete transition and load iframe content
                        completeContentMorphing(component, sourceCard);
                    });
                });
            });
        }

        function extractContentElements(sourceCard) {
            const cardIcon = sourceCard.querySelector('.card-icon i');
            const cardTitle = sourceCard.querySelector('.card-title');
            const cardDescription = sourceCard.querySelector('.card-description');

            const elements = {};

            if (cardIcon) {
                const iconRect = cardIcon.getBoundingClientRect();
                elements.icon = createMorphingElement(cardIcon, iconRect, 'morphing-icon');
            }

            if (cardTitle) {
                const titleRect = cardTitle.getBoundingClientRect();
                elements.title = createMorphingElement(cardTitle, titleRect, 'morphing-title');
            }

            if (cardDescription) {
                const descRect = cardDescription.getBoundingClientRect();
                elements.description = createMorphingElement(cardDescription, descRect, 'morphing-description');
            }

            return elements;
        }

        function createMorphingElement(originalElement, rect, className) {
            const morphElement = originalElement.cloneNode(true);
            
            // Position exactly over original
            morphElement.style.position = 'fixed';
            morphElement.style.top = rect.top + 'px';
            morphElement.style.left = rect.left + 'px';
            morphElement.style.width = rect.width + 'px';
            morphElement.style.height = rect.height + 'px';
            morphElement.style.zIndex = '1002';
            morphElement.style.pointerEvents = 'none';
            morphElement.style.margin = '0';
            morphElement.classList.add(className);

            // Copy computed styles for accurate morphing
            const computedStyles = window.getComputedStyle(originalElement);
            morphElement.style.fontSize = computedStyles.fontSize;
            morphElement.style.fontWeight = computedStyles.fontWeight;
            morphElement.style.color = computedStyles.color;
            morphElement.style.backgroundColor = computedStyles.backgroundColor;

            document.body.appendChild(morphElement);
            return morphElement;
        }

        function getModalTargetPositions() {
            const modalContainer = modal.querySelector('.component-modal-container');
            if (!modalContainer) return {};

            // Get actual DOM elements from modal header
            const modalIcon = modal.querySelector('.modal-component-icon');
            const modalTitle = modal.querySelector('.modal-title');
            const modalSubtitle = modal.querySelector('.modal-subtitle');
            
            const positions = {};

            if (modalIcon) {
                const iconRect = modalIcon.getBoundingClientRect();
                positions.icon = {
                    top: iconRect.top,
                    left: iconRect.left,
                    width: iconRect.width,
                    height: iconRect.height
                };
            }

            if (modalTitle) {
                const titleRect = modalTitle.getBoundingClientRect();
                positions.title = {
                    top: titleRect.top,
                    left: titleRect.left,
                    width: titleRect.width,
                    height: titleRect.height
                };
            }

            if (modalSubtitle) {
                const subtitleRect = modalSubtitle.getBoundingClientRect();
                positions.description = {
                    top: subtitleRect.top,
                    left: subtitleRect.left,
                    width: subtitleRect.width,
                    height: subtitleRect.height
                };
            }

            return positions;
        }

        function morphContentElements(elements, targetPositions, onComplete) {
            const morphDuration = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--content-morph-duration').replace('s', '')) * 1000;
            let completedMorphs = 0;
            const totalMorphs = Object.keys(elements).length;

            Object.keys(elements).forEach((key, index) => {
                const element = elements[key];
                const target = targetPositions[key];
                
                if (!element || !target) {
                    completedMorphs++;
                    if (completedMorphs === totalMorphs && onComplete) onComplete();
                    return;
                }

                // Add slight delay for staggered effect
                setTimeout(() => {
                    element.style.top = target.top + 'px';
                    element.style.left = target.left + 'px';
                    element.style.width = target.width + 'px';
                    element.style.height = target.height + 'px';

                    // Adjust styling for target context
                    if (key === 'icon') {
                        element.style.borderRadius = '0.75rem';
                        element.style.background = 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)';
                        element.style.color = 'white';
                        element.style.display = 'flex';
                        element.style.alignItems = 'center';
                        element.style.justifyContent = 'center';
                        element.style.fontSize = '1.25rem';
                    } else if (key === 'title') {
                        element.style.fontSize = '1.5rem';
                        element.style.fontWeight = '700';
                        element.style.color = '#0f766e';
                    } else if (key === 'description') {
                        element.style.fontSize = '0.875rem';
                        element.style.color = '#64748b';
                        element.style.opacity = '0.8';
                    }

                    setTimeout(() => {
                        completedMorphs++;
                        if (completedMorphs === totalMorphs && onComplete) onComplete();
                    }, morphDuration);
                }, index * 100);
            });

            // Fallback timeout
            setTimeout(() => {
                if (completedMorphs < totalMorphs && onComplete) onComplete();
            }, morphDuration + 500);
        }

        function createModalFrame(targetPositions, onComplete) {
            const modal = mainContentContainer.querySelector('#componentModal');
            const modalContainer = modal.querySelector('.component-modal-container');
            
            if (!modalContainer) {
                if (onComplete) onComplete();
                return;
            }

            const containerRect = modalContainer.getBoundingClientRect();
            
            // Create frame element
            const frame = document.createElement('div');
            frame.className = 'modal-frame';
            frame.style.top = containerRect.top + 'px';
            frame.style.left = containerRect.left + 'px';
            frame.style.width = containerRect.width + 'px';
            frame.style.height = containerRect.height + 'px';
            
            document.body.appendChild(frame);

            // Trigger growing animation
            requestAnimationFrame(() => {
                frame.classList.add('growing');
                
                const frameDuration = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--frame-grow-duration').replace('s', '')) * 1000;
                
                setTimeout(() => {
                    if (onComplete) onComplete();
                }, frameDuration);
            });
        }

        function createVisualBridge(sourceCard) {
            const cardRect = sourceCard.getBoundingClientRect();
            const particles = 8;
            
            // Create connection particles
            for (let i = 0; i < particles; i++) {
                setTimeout(() => {
                    const particle = document.createElement('div');
                    particle.className = 'bridge-particle';
                    particle.style.left = (cardRect.left + cardRect.width / 2) + 'px';
                    particle.style.top = (cardRect.top + cardRect.height / 2) + 'px';
                    
                    document.body.appendChild(particle);
                    
                    // Remove particle after animation
                    setTimeout(() => {
                        if (particle.parentNode) {
                            document.body.removeChild(particle);
                        }
                    }, 600);
                }, i * 50);
            }

            // Create connection line
            const line = document.createElement('div');
            line.className = 'connection-line';
            line.style.left = cardRect.left + 'px';
            line.style.top = (cardRect.top + cardRect.height / 2) + 'px';
            line.style.width = window.innerWidth - cardRect.left + 'px';
            
            document.body.appendChild(line);
            
            // Remove line after animation
            setTimeout(() => {
                if (line.parentNode) {
                    document.body.removeChild(line);
                }
            }, 600);

            // Create glow effect
            const glow = document.createElement('div');
            glow.className = 'morphing-glow';
            glow.style.left = (cardRect.left + cardRect.width / 2 - 50) + 'px';
            glow.style.top = (cardRect.top + cardRect.height / 2 - 50) + 'px';
            glow.style.width = '100px';
            glow.style.height = '100px';
            
            document.body.appendChild(glow);
            
            // Remove glow after animation
            setTimeout(() => {
                if (glow.parentNode) {
                    document.body.removeChild(glow);
                }
            }, 600);
        }

        function hideModalHeaderElements() {
            const modalIcon = modal.querySelector('.modal-component-icon');
            const modalTitle = modal.querySelector('.modal-title');
            const modalSubtitle = modal.querySelector('.modal-subtitle');

            if (modalIcon) modalIcon.style.opacity = '0';
            if (modalTitle) modalTitle.style.opacity = '0';
            if (modalSubtitle) modalSubtitle.style.opacity = '0';
        }

        function showModalHeaderElements() {
            const modalIcon = modal.querySelector('.modal-component-icon');
            const modalTitle = modal.querySelector('.modal-title');
            const modalSubtitle = modal.querySelector('.modal-subtitle');

            if (modalIcon) {
                modalIcon.style.opacity = '1';
                modalIcon.style.transition = 'opacity 0.3s ease';
            }
            if (modalTitle) {
                modalTitle.style.opacity = '1';
                modalTitle.style.transition = 'opacity 0.3s ease';
            }
            if (modalSubtitle) {
                modalSubtitle.style.opacity = '1';
                modalSubtitle.style.transition = 'opacity 0.3s ease';
            }
        }

        function completeContentMorphing(component, sourceCard) {
            const modalContainer = modal.querySelector('.component-modal-container');
            if (modalContainer) {
                modalContainer.classList.add('content-ready');
            }

            // Fade out morphing elements before removing them
            const morphingElements = document.querySelectorAll('.morphing-icon, .morphing-title, .morphing-description');
            morphingElements.forEach(el => {
                el.style.transition = 'opacity 0.4s ease';
                el.style.opacity = '0';
            });

            // Show modal header elements with smooth transition
            showModalHeaderElements();

            // Remove morphing elements after fade out
            setTimeout(() => {
                const allMorphingElements = document.querySelectorAll('.morphing-icon, .morphing-title, .morphing-description, .modal-frame');
                allMorphingElements.forEach(el => {
                    if (el.parentNode) {
                        el.parentNode.removeChild(el);
                    }
                });

                // Restore source card
                sourceCard.classList.remove('content-morphing');

                // Load iframe content with enhanced loading
                loadComponentWithEnhancements(component);
            }, 400);
        }

        function openComponentModal(index) {
            if (index < 0 || index >= components.length) return;

            currentComponentIndex = index;
            const component = components[index];
            
            // Update modal content
            updateModalContent(component);
            
            // Show modal
            modal.classList.add('show', 'component-modal');
            document.body.style.overflow = 'hidden';
            
            // Load component into iframe
            loadComponentWithEnhancements(component);
            
            // Update navigation buttons
            updateNavButtons();
        }

        // --- Enhanced Modal Utility Functions ---
        function updateModalContent(component) {
            if (modalTitle) modalTitle.textContent = component.title;
            if (modalSubtitle) modalSubtitle.textContent = component.subtitle;
            if (modalIcon) modalIcon.className = component.icon;
        }

        function loadComponentWithEnhancements(component) {
            console.log('🔄 loadComponentWithEnhancements called for:', component.id);
            if (!iframe || !loadingDiv || !comingSoonDiv) {
                console.error('❌ Missing iframe elements:', {
                    iframe: !!iframe,
                    loadingDiv: !!loadingDiv,
                    comingSoonDiv: !!comingSoonDiv
                });
                return;
            }
            console.log('✅ All iframe elements found, proceeding with load');

            // Check if component is preloaded
            const preloaded = modalSystem.preloadCache.get(component.id);
            console.log('🔍 Preload check for:', component.id, 'preloaded:', !!preloaded);
            
            if (preloaded) {
                const timeDiff = Date.now() - preloaded.loaded;
                console.log('⏰ Preload time difference:', timeDiff, 'ms (5min cache = 300000ms)');
                
                if (timeDiff < 300000) { // 5 minute cache
                    // VALIDATION: Check preloaded iframe src is valid
                    console.log('🔍 VALIDATION: Checking preloaded iframe src...');
                    console.log('🔍 VALIDATION: preloaded.iframe exists:', !!preloaded.iframe);
                    console.log('🔍 VALIDATION: preloaded.iframe.src:', preloaded.iframe?.src);
                    console.log('🔍 VALIDATION: src is not empty:', preloaded.iframe?.src && preloaded.iframe.src.trim() !== '');
                    
                    if (preloaded.iframe && preloaded.iframe.src && preloaded.iframe.src.trim() !== '') {
                        console.log('✅ VALIDATION PASSED: Using preloaded content for:', component.id);
                        showPreloadedContent(preloaded.iframe, component);
                        return; // Critical: return here to prevent fallthrough
                    } else {
                        console.log('❌ VALIDATION FAILED: Preloaded content has invalid src, loading normally');
                        console.log('❌ VALIDATION FAILED: Invalid src value:', preloaded.iframe?.src);
                    }
                } else {
                    console.log('⏰ Preloaded content expired, loading normally');
                }
            } else {
                console.log('📥 No preloaded content found, loading normally');
            }
            
            // Load normally with enhanced loading state
            console.log('🌐 Calling loadComponentNormally for:', component.id);
            loadComponentNormally(component);
        }

        function showPreloadedContent(preloadedIframe, component) {
            console.log('⚡ showPreloadedContent called for:', component.id);
            console.log('⚡ preloadedIframe.src:', preloadedIframe.src);
            
            // Enhanced loading display for preloaded content
            loadingDiv.style.display = 'flex';
            loadingDiv.innerHTML = `
                <div class="loading-spinner-enhanced">
                    <div class="spinner"></div>
                    <div class="loading-text">⚡ Loading preloaded content...</div>
                </div>
            `;
            comingSoonDiv.classList.add('hidden');
            iframe.style.display = 'none';
            iframe.classList.remove('loaded');
            
            console.log('⚡ Starting preloaded content transition...');
            
            // Quick transition to loaded content
            setTimeout(() => {
                console.log('⚡ Setting iframe src to preloaded content');
                iframe.src = preloadedIframe.src;
                iframe.onload = () => {
                    console.log('✅ Preloaded iframe loaded successfully for:', component.id);
                    loadingDiv.classList.add('fade-out');
                    iframe.classList.add('loaded');
                    setTimeout(() => {
                        loadingDiv.style.display = 'none';
                        iframe.style.display = 'block';
                        loadingDiv.classList.remove('fade-out');
                        console.log('🎉 Preloaded iframe display completed for:', component.id);
                    }, 400);
                };
                iframe.onerror = () => {
                    console.error('❌ Preloaded iframe failed to load:', preloadedIframe.src);
                    loadingDiv.style.display = 'none';
                    iframe.style.display = 'none';
                    comingSoonDiv.classList.remove('hidden');
                };
            }, 200);
        }

        function loadComponentNormally(component) {
            // Enhanced loading state
            loadingDiv.style.display = 'flex';
            loadingDiv.innerHTML = `
                <div class="loading-spinner-enhanced">
                    <div class="spinner"></div>
                    <div class="loading-text">Loading ${component.title}...</div>
                </div>
            `;
            comingSoonDiv.classList.add('hidden');
            iframe.style.display = 'none';
            iframe.classList.remove('loaded');
            iframe.src = 'about:blank';

            if (component.available && component.iframe) {
                console.log('🌐 Loading iframe:', component.iframe);
                setTimeout(() => {
                    iframe.onload = () => {
                        console.log('✅ Iframe loaded successfully for:', component.id);
                        loadingDiv.classList.add('fade-out');
                        setTimeout(() => {
                            loadingDiv.style.display = 'none';
                            iframe.style.display = 'block';
                            iframe.classList.add('loaded');
                            loadingDiv.classList.remove('fade-out');
                            console.log('🎉 Iframe display completed for:', component.id);
                        }, 400);
                    };
                    iframe.onerror = () => {
                        console.error('❌ Iframe failed to load:', component.iframe);
                        loadingDiv.style.display = 'none';
                        iframe.style.display = 'none';
                        comingSoonDiv.classList.remove('hidden');
                    };
                    
                    console.log('🔗 Setting iframe.src to:', component.iframe);
                    iframe.src = component.iframe;
                }, 100);
            } else {
                console.log('⚠️ Component not available, showing coming soon');
                loadingDiv.style.display = 'none';
                comingSoonDiv.classList.remove('hidden');
            }
        }
        
        /**
         * LEGACY: Original loading function for compatibility
         * @param {object} component - The component object from the components array.
         */
        function loadComponent(component) {
            if (!iframe || !loadingDiv || !comingSoonDiv) {
                console.error('Legacy loadComponent missing elements:', {
                    iframe: !!iframe,
                    loadingDiv: !!loadingDiv,
                    comingSoonDiv: !!comingSoonDiv
                });
                return;
            }

            // 1. Reset iframe and show loading state
            loadingDiv.style.display = 'flex';
            comingSoonDiv.classList.add('hidden');
            iframe.style.display = 'none';
            iframe.src = 'about:blank'; // Important: Clear old content and stop scripts

            if (component.available && component.iframe) {
                // 2. Use a tiny timeout to ensure the DOM has updated (good practice for modals)
                setTimeout(() => {
                    // 3. Set up event handlers *before* setting the new src
                    iframe.onload = () => {
                        loadingDiv.style.display = 'none';
                        iframe.style.display = 'block';
                    };
                    iframe.onerror = () => {
                        loadingDiv.style.display = 'none';
                        iframe.style.display = 'none';
                        comingSoonDiv.classList.remove('hidden');
                    };
                    
                    // 4. Set the new source to trigger loading
                    iframe.src = component.iframe;
                }, 50);

            } else {
                // Show "Coming Soon" for unavailable components
                loadingDiv.style.display = 'none';
                comingSoonDiv.classList.remove('hidden');
            }
        }
        
        function updateNavButtons() {
            if (prevBtn) prevBtn.disabled = currentComponentIndex === 0;
            if (nextBtn) nextBtn.disabled = currentComponentIndex === components.length - 1;
        }
        
        function navigateComponent(direction) {
            const newIndex = currentComponentIndex + direction;
            openComponentModal(newIndex);
        }
        
        function closeModal() {
            modal.classList.remove('show', 'component-modal');
            document.body.style.overflow = '';
            
            // Clear iframe to stop any ongoing processes
            if (iframe) {
                iframe.src = 'about:blank';
                iframe.classList.remove('loaded');
            }
            
            // Clean up all morphing and bridge elements
            const morphingElements = document.querySelectorAll(
                '.morphing, .morphing-complete, .morphing-icon, .morphing-title, .morphing-description, ' +
                '.modal-frame, .bridge-particle, .connection-line, .morphing-glow'
            );
            morphingElements.forEach(el => {
                if (el.parentNode === document.body) {
                    document.body.removeChild(el);
                }
            });
            
            // Reset modal container state
            const modalContainer = modal.querySelector('.component-modal-container');
            if (modalContainer) {
                modalContainer.classList.remove('content-ready');
            }

            // Reset modal header elements opacity
            const modalIcon = modal.querySelector('.modal-component-icon');
            const modalTitle = modal.querySelector('.modal-title');
            const modalSubtitle = modal.querySelector('.modal-subtitle');
            
            if (modalIcon) modalIcon.style.opacity = '';
            if (modalTitle) modalTitle.style.opacity = '';
            if (modalSubtitle) modalSubtitle.style.opacity = '';
            
            // Restore card states
            const cards = mainContentContainer.querySelectorAll('.component-card');
            cards.forEach(card => {
                card.style.opacity = '';
                card.classList.remove('content-morphing');
            });
        }

        // --- Event Handlers for Component Modal ---
        if (closeBtn) closeBtn.addEventListener('click', closeModal);
        if (prevBtn) prevBtn.addEventListener('click', () => navigateComponent(-1));
        if (nextBtn) nextBtn.addEventListener('click', () => navigateComponent(1));
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });
        
        document.addEventListener('keydown', (e) => {
            if (modal.classList.contains('show')) {
                switch(e.key) {
                    case 'Escape':
                        closeModal();
                        break;
                    case 'ArrowLeft':
                        if (prevBtn && !prevBtn.disabled) navigateComponent(-1);
                        break;
                    case 'ArrowRight':
                        if (nextBtn && !nextBtn.disabled) navigateComponent(1);
                        break;
                }
            }
        });
    }

    function initializeAuthorsBox() {
        const authorsBox = mainContentContainer.querySelector('#authorsBox');
        const modal = mainContentContainer.querySelector('#authorsModal');
        const closeBtn = mainContentContainer.querySelector('#closeModal');
        
        if (authorsBox && modal) {
            const closeModal = () => {
                modal.classList.remove('show', 'authors-modal');
                document.body.style.overflow = '';
            };

            authorsBox.addEventListener('click', () => {
                modal.classList.add('show', 'authors-modal');
                document.body.style.overflow = 'hidden';
            });
            
            if (closeBtn) closeBtn.addEventListener('click', closeModal);
            modal.addEventListener('click', (e) => {
                if (e.target === modal) closeModal();
            });
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && modal.classList.contains('show')) closeModal();
            });
        }
    }

    // --- Dynamic Cursor Tracking for Modal Backgrounds ---
    function initializeCursorTracking() {
        let mouseMoveTimeout;
        
        document.addEventListener('mousemove', (e) => {
            clearTimeout(mouseMoveTimeout);
            
            const modal = document.querySelector('.modal-overlay.show');
            if (modal) {
                const x = (e.clientX / window.innerWidth) * 100;
                const y = (e.clientY / window.innerHeight) * 100;
                
                modal.style.setProperty('--cursor-x', `${x}%`);
                modal.style.setProperty('--cursor-y', `${y}%`);
            }
            
            // Throttle updates for performance
            mouseMoveTimeout = setTimeout(() => {
                // Additional cursor-based effects can go here
            }, 16); // ~60fps
        });
    }

    // --- Load Tab Content with Enhanced Loading ---
    async function loadTabContent(dataSourceUrl, targetTabId) {
        const strategy = getLoadingStrategy();
        const startTime = performance.now();
        
        try {
            // Show skeleton loading state immediately
            showLoadingState(targetTabId);
            
            // Add minimum loading time to prevent flash
            const [response] = await Promise.all([
                fetch(dataSourceUrl),
                new Promise(resolve => setTimeout(resolve, strategy.skeletonDuration))
            ]);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const html = await response.text();
            
            // Fade out skeleton loader
            const loadingContainer = document.getElementById('loadingContainer');
            if (loadingContainer) {
                loadingContainer.classList.add('fade-out');
                
                // Wait for fade-out animation, then replace content
                setTimeout(() => {
                    mainContentContainer.innerHTML = html;
                    
                    // Apply staggered loading animations
                    applyStaggeredLoading();
                    
                    // Initialize MathJax if available
                    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
                        MathJax.typesetPromise([mainContentContainer]);
                    }
                    
                    // Disconnect observers
                    if (contentSectionObserver) contentSectionObserver.disconnect();
                    if (apiSectionObserver) apiSectionObserver.disconnect();

                    // Tab-specific setup with adaptive loading
                    if (targetTabId === 'example' || targetTabId === 'api') {
                        setupContentSectionObserver(); 
                        
                        // Apply visibility with network-aware delays
                        const sections = mainContentContainer.querySelectorAll('.fade-in-section, .api-section-content');
                        sections.forEach((sec, index) => {
                            setTimeout(() => {
                                if (isElementInViewport(sec)) sec.classList.add('visible');
                            }, index * strategy.staggerDelay);
                        });
                    }
                    
                    // Playground tab specific setup
                    if (targetTabId === 'playground') {
                        console.log('🎮 Setting up playground tab');
                        // Add small delay to ensure DOM is fully ready
                        setTimeout(() => {
                            console.log('⏰ First initialization attempt (100ms delay)');
                            initializeComponentDiscovery();
                            initializeAuthorsBox();
                        }, 100);
                        
                        // Fallback check in case first attempt fails
                        setTimeout(() => {
                            console.log('🔄 Fallback check (500ms delay)');
                            const componentCards = mainContentContainer.querySelectorAll('.component-card');
                            const modal = mainContentContainer.querySelector('#componentModal');
                            console.log('🔍 Fallback check results:', {
                                componentCards: componentCards.length,
                                modal: !!modal,
                                hasEventListeners: modal ? modal.hasEventListeners : 'no modal'
                            });
                            if (componentCards.length && modal && !modal.hasEventListeners) {
                                console.log('🆘 Fallback initialization triggered');
                                initializeComponentDiscovery();
                            }
                        }, 500);
                    } else if (targetTabId === 'api') {
                        initializeApiTabElements();
                        setupApiTabObserver();
                    }
                    
                    // Optimize for slow networks
                    if (strategy.enableLazyLoading) {
                        implementLazyLoading();
                    }
                    
                    // Reduce animations for slow networks
                    if (strategy.reduceAnimations) {
                        document.body.classList.add('reduce-motion');
                    }
                    
                }, 300);
            } else {
                // Fallback if skeleton loader is not found
                mainContentContainer.innerHTML = html;
                applyStaggeredLoading();
            }
            
            // Performance logging
            const loadTime = performance.now() - startTime;
            console.log(`Content loaded in ${loadTime.toFixed(2)}ms with ${strategy.skeletonDuration}ms skeleton duration`);
            
        } catch (error) {
            console.error("Error loading tab content:", error);
            
            // Show error state with retry functionality
            const retryFunction = () => loadTabContent(dataSourceUrl, targetTabId);
            showErrorState(error, retryFunction);
            
            // Track error for potential network quality adjustment
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                console.warn('Network error detected, consider adjusting loading strategy');
            }
        }
    }

    // --- Lazy Loading Implementation ---
    function implementLazyLoading() {
        const images = mainContentContainer.querySelectorAll('img[src]');
        const videos = mainContentContainer.querySelectorAll('video[src]');
        
        const lazyLoadObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const media = entry.target;
                    if (media.dataset.src) {
                        media.src = media.dataset.src;
                        media.removeAttribute('data-src');
                    }
                    lazyLoadObserver.unobserve(media);
                }
            });
        }, {
            rootMargin: '50px'
        });

        // Convert src to data-src for lazy loading
        [...images, ...videos].forEach(media => {
            if (media.src && !media.dataset.src) {
                media.dataset.src = media.src;
                media.removeAttribute('src');
                lazyLoadObserver.observe(media);
            }
        });
    }

    // --- Primary Tab Switching Logic ---
    primaryTabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTabId = button.dataset.tab;
            const dataSourceUrl = button.dataset.source;

            primaryTabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            loadTabContent(dataSourceUrl, targetTabId).then(() => {
                 currentStickyHeaderHeight = stickyHeader ? stickyHeader.offsetHeight : 0; 
                 window.scrollTo({ top: 0, behavior: 'auto' }); 
            });
        });
    });

    // --- Initialize API Tab Specific Elements and Event Listeners ---
    function initializeApiTabElements() {
        apiNavLinks = mainContentContainer.querySelectorAll('#apiNavMenu .api-nav-link');
        apiRightPane = mainContentContainer.querySelector('#apiRightPane');
        apiContentSections = mainContentContainer.querySelectorAll('#apiContent .api-section-content'); 

        if (apiNavLinks.length > 0 && apiRightPane) {
            apiNavLinks.forEach(link => {
                link.removeEventListener('click', handleApiNavClick); 
                link.addEventListener('click', handleApiNavClick);
            });
        }
    }

    function handleApiNavClick(e) { 
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        const targetElement = mainContentContainer.querySelector(`#${targetId}`); 

        apiNavLinks.forEach(nav => nav.classList.remove('active-api-link'));
        this.classList.add('active-api-link');

        if (targetElement && apiRightPane) {
            const paneTop = apiRightPane.getBoundingClientRect().top;
            const targetTopInPane = targetElement.getBoundingClientRect().top;
            const scrollToPosition = apiRightPane.scrollTop + (targetTopInPane - paneTop) - 10; 

            apiRightPane.scrollTo({
                top: scrollToPosition,
                behavior: 'smooth'
            });
        }
    }

    // --- Scroll to Top Button Functionality ---
    if (scrollToTopBtn) {
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 100) {
                if (scrollToTopBtn.style.display !== "flex") {
                    scrollToTopBtn.style.display = "flex";
                    requestAnimationFrame(() => { scrollToTopBtn.style.opacity = "1"; });
                }
            } else {
                if (scrollToTopBtn.style.opacity === "1") {
                    scrollToTopBtn.style.opacity = "0";
                    setTimeout(() => {
                        if (scrollToTopBtn.style.opacity === "0") scrollToTopBtn.style.display = "none";
                    }, 300);
                }
            }
        });
        scrollToTopBtn.addEventListener('click', function() {
            window.scrollTo({top: 0, behavior: 'smooth'});
        });
    }

    // --- Intersection Observer for Content Sections (Fade-in) ---
    function setupContentSectionObserver() {
        if (contentSectionObserver) contentSectionObserver.disconnect(); 

        const allContentSectionsForObserver = mainContentContainer.querySelectorAll('.content-section.fade-in-section');
      
        const observerCallback = (entries) => {
            entries.forEach(entry => {
                if (!mainContentContainer.contains(entry.target)) return;
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        };
        
        const observerOptions = {
            root: null, 
            rootMargin: `-${currentStickyHeaderHeight + 24}px 0px -40% 0px`, 
            threshold: 0.01 
        };
        contentSectionObserver = new IntersectionObserver(observerCallback, observerOptions);

        allContentSectionsForObserver.forEach(section => { 
            if (section) contentSectionObserver.observe(section); 
        });
    }

    // --- Intersection Observer for API Tab ---
    function setupApiTabObserver() {
        if (apiSectionObserver) apiSectionObserver.disconnect();
        const scrollablePane = mainContentContainer.querySelector('#apiRightPane'); 
        const sectionsToObserve = mainContentContainer.querySelectorAll('#apiRightPane .content-section, #apiRightPane .api-section-content');

        if (!scrollablePane || sectionsToObserve.length === 0) {
            return;
        }
        apiRightPane = scrollablePane; 
        apiContentSections = sectionsToObserve;

        const apiObserverOptions = {
            root: apiRightPane, 
            rootMargin: "-20px 0px -60% 0px", 
            threshold: 0.01 
        };

        let lastActiveApiLink = null;

        apiSectionObserver = new IntersectionObserver(entries => {
            const activePrimaryTab = document.querySelector('#primaryTabsNav .tab-button.active');
            if (!activePrimaryTab || activePrimaryTab.dataset.tab !== 'api') {
                return; 
            }

            let bestVisibleEntry = null;
            entries.forEach(entry => {
                 if (!apiRightPane.contains(entry.target)) return; 

                if (entry.target.classList.contains('fade-in-section') || entry.target.classList.contains('api-section-content')) {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                    }
                }

                if (entry.isIntersecting) {
                    if (!bestVisibleEntry || entry.boundingClientRect.top < bestVisibleEntry.boundingClientRect.top) {
                        bestVisibleEntry = entry;
                    }
                }
            });
            
            const currentApiNavLinks = mainContentContainer.querySelectorAll('#apiNavMenu .api-nav-link'); 

            if (bestVisibleEntry) {
                const id = bestVisibleEntry.target.id;
                const correspondingLink = mainContentContainer.querySelector(`.api-nav-link[href="#${id}"]`);
                if (correspondingLink && correspondingLink !== lastActiveApiLink) {
                    currentApiNavLinks.forEach(nav => nav.classList.remove('active-api-link'));
                    correspondingLink.classList.add('active-api-link');
                    lastActiveApiLink = correspondingLink;
                }
            } else if (!entries.some(e => e.isIntersecting) && lastActiveApiLink && apiRightPane.scrollTop === 0 && currentApiNavLinks.length > 0) {
                currentApiNavLinks.forEach(nav => nav.classList.remove('active-api-link'));
                currentApiNavLinks[0].classList.add('active-api-link');
                lastActiveApiLink = currentApiNavLinks[0];
            }
        }, apiObserverOptions);

        apiContentSections.forEach(section => {
            if(section) apiSectionObserver.observe(section);
        });
        
        const activePrimaryTab = document.querySelector('#primaryTabsNav .tab-button.active');
        if (activePrimaryTab && activePrimaryTab.dataset.tab === 'api') {
            setTimeout(() => { 
                if(apiRightPane && apiRightPane.scrollTop === 0 && mainContentContainer.querySelectorAll('#apiNavMenu .api-nav-link').length > 0) {
                    const currentApiNavLinks = mainContentContainer.querySelectorAll('#apiNavMenu .api-nav-link');
                     if(currentApiNavLinks.length > 0 && !lastActiveApiLink){ 
                        currentApiNavLinks.forEach(nav => nav.classList.remove('active-api-link'));
                        currentApiNavLinks[0].classList.add('active-api-link');
                        lastActiveApiLink = currentApiNavLinks[0];
                     }
                }
                 if(apiRightPane) { apiRightPane.scrollTop +=1; apiRightPane.scrollTop -=1; }
            }, 150); 
        }
    }

    // --- Event Listener for Hero CTA Buttons (delegated for main tab switching) ---
    document.addEventListener('click', function(event) {
        const ctaTabButton = event.target.closest('.hero-cta-button[data-tab-target]');
        if (ctaTabButton) {
            event.preventDefault();
            const targetTabId = ctaTabButton.dataset.tabTarget;
            const mainTabButton = document.querySelector(`#primaryTabsNav .tab-button[data-tab="${targetTabId}"]`);
            if (mainTabButton) {
                mainTabButton.click(); 
            }
            return; 
        }
    });

    // --- Remove Initial Loading State ---
    function hideInitialLoadingState() {
        const initialLoader = document.getElementById('initialLoadingState');
        if (initialLoader) {
            initialLoader.style.display = 'none';
        }
    }

    // --- Initial Load ---
    const initialTabButton = document.querySelector('#primaryTabsNav .tab-button.active');
    if (initialTabButton) {
        hideInitialLoadingState();
        loadTabContent(initialTabButton.dataset.source, initialTabButton.dataset.tab);
    } else {
        const firstTabButton = document.querySelector('#primaryTabsNav .tab-button');
        if (firstTabButton) {
            firstTabButton.classList.add('active');
            hideInitialLoadingState();
            loadTabContent(firstTabButton.dataset.source, firstTabButton.dataset.tab);
        }
    }

    // --- Connection Quality Monitoring ---
    if (navigator.connection) {
        navigator.connection.addEventListener('change', () => {
            const newQuality = detectNetworkQuality();
            document.body.className = document.body.className.replace(/\b\w+-network\b/g, '');
            document.body.classList.add(`${newQuality}-network`);
            console.log(`Network quality changed to: ${newQuality}`);
        });
    }

    // --- Performance Monitoring ---
    if ('PerformanceObserver' in window) {
        const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach(entry => {
                if (entry.entryType === 'navigation') {
                    const loadTime = entry.loadEventEnd - entry.loadEventStart;
                    if (loadTime > 3000) {
                        console.warn('Slow page load detected, consider optimizations');
                        document.body.classList.add('slow-loading');
                    }
                }
            });
        });
        
        observer.observe({ entryTypes: ['navigation'] });
    }

    // --- Title Click Functionality ---
    // Add click handler to the main title to navigate to playground
    const mainTitle = document.querySelector('.sticky-header .flex-col.sm\\:flex-row a');
    if (mainTitle) {
        mainTitle.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Find and click the playground tab button
            const playgroundButton = document.querySelector('#primaryTabsNav .tab-button[data-tab="playground"]');
            if (playgroundButton) {
                // Remove active class from all tabs
                primaryTabButtons.forEach(btn => btn.classList.remove('active'));
                
                // Add active class to playground tab
                playgroundButton.classList.add('active');
                
                // Load playground content
                loadTabContent(playgroundButton.dataset.source, playgroundButton.dataset.tab).then(() => {
                    currentStickyHeaderHeight = stickyHeader ? stickyHeader.offsetHeight : 0; 
                    window.scrollTo({ top: 0, behavior: 'auto' }); 
                });
            }
        });
        
        // Add visual feedback for the clickable title
        mainTitle.style.cursor = 'pointer';
        mainTitle.addEventListener('mouseenter', () => {
            mainTitle.style.transform = 'scale(1.02)';
        });
        mainTitle.addEventListener('mouseleave', () => {
            mainTitle.style.transform = 'scale(1)';
        });
    }
});
