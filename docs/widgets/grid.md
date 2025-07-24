# Grid Effects and Smooth Transitions Documentation

This document provides a comprehensive guide to implementing smooth grid effects and transitions in visualization widgets. Each section can be implemented independently or combined for richer interactions.

## Core Principles

### 1. Element Reuse Strategy
**Problem**: Clearing `innerHTML = ''` and recreating DOM elements prevents CSS transitions.
**Solution**: Reuse existing elements and update their properties.

```javascript
// ❌ BAD - Destroys transitions
container.innerHTML = '';
data.forEach(item => {
    const element = document.createElement('div');
    // ... set properties
    container.appendChild(element);
});

// ✅ GOOD - Preserves transitions  
const existingElements = Array.from(container.children);
data.forEach((item, index) => {
    let element = existingElements[index];
    if (!element) {
        element = document.createElement('div');
        element.className = 'target-class';
        container.appendChild(element);
    }
    // ... update properties
});

// Remove extra elements
while (container.children.length > data.length) {
    container.lastChild.remove();
}
```

### 2. CSS Transition Requirements
**Essential**: Include position properties in CSS transitions.

```css
.animated-element {
    position: absolute;
    transition: left 0.4s var(--ease-out-cubic), 
                bottom 0.4s var(--ease-out-cubic), 
                width 0.4s var(--ease-out-cubic),
                height 0.4s var(--ease-out-cubic),
                transform 0.25s var(--ease-out-cubic);
    will-change: left, bottom, transform;
}
```

## Component-Specific Implementations

### 1. Mapping Windows (Green/Teal Boxes)

**Visual**: Green/teal bordered rectangles showing ideal mapping regions
**Transition**: Position and size changes when test cases change

#### CSS Requirements
```css
.ideal-mapping-window { 
    position: absolute; 
    border: 2px solid #0d9488; 
    background-color: rgba(20, 184, 166, 0.3); 
    box-sizing: border-box; 
    z-index: 1; 
    transition: all 0.3s var(--ease-out-quad); 
    pointer-events: none; 
    border-radius: 0.375rem; 
    box-shadow: 0 0 8px rgba(13, 148, 136, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.2);
}
```

#### JavaScript Implementation
```javascript
function updateMappingWindows(chartUI, idealWindows, sourceLen, targetLen) {
    const existingWindows = Array.from(chartUI.idealContainer.children);
    
    idealWindows.forEach((window, x_idx) => {
        if (!window) return;
        
        let windowDiv = existingWindows[x_idx];
        if (!windowDiv) {
            windowDiv = document.createElement('div');
            windowDiv.className = 'ideal-mapping-window';
            chartUI.idealContainer.appendChild(windowDiv);
        }
        
        // Update position and size - CSS transitions handle the animation
        windowDiv.style.left = `calc(${x_idx} * 100% / ${sourceLen})`;
        windowDiv.style.width = `calc(100% / ${sourceLen})`;
        windowDiv.style.bottom = `calc(${window.start} * 100% / ${targetLen})`;
        windowDiv.style.height = `calc(${(window.end - window.start)} * 100% / ${targetLen})`;
    });
    
    // Remove extra windows
    while (chartUI.idealContainer.children.length > idealWindows.length) {
        chartUI.idealContainer.lastChild.remove();
    }
}
```

### 2. LCT Padding Windows (Yellow Dashed Boxes)

**Visual**: Yellow dashed rectangles showing LCT tolerance zones
**Transition**: Size and position changes when LCT slider moves

#### CSS Requirements
```css
.lct-padding-window { 
    position: absolute; 
    background-color: rgba(253, 224, 71, 0.15); 
    box-sizing: border-box; 
    z-index: 0; 
    transition: all 0.2s ease-out; 
    pointer-events: none; 
    border: 1px dashed #facc15; 
    border-radius: 0.25rem; 
}
```

#### JavaScript Implementation
```javascript
function updateLctWindows(chartUI, idealWindows, currentLct, sourceLen, targetLen) {
    const existingLctWindows = Array.from(chartUI.lctContainer.children);
    
    idealWindows.forEach((window, x_idx) => {
        if (!window) return;
        
        let lctDiv = existingLctWindows[x_idx];
        if (!lctDiv) {
            lctDiv = document.createElement('div');
            lctDiv.className = 'lct-padding-window';
            chartUI.lctContainer.appendChild(lctDiv);
        }
        
        const lctStartY = Math.max(0, window.start - currentLct);
        const lctEndY = Math.min(targetLen, window.end + currentLct);
        const lctHeight = lctEndY - lctStartY;
        
        lctDiv.style.left = `calc(${x_idx} * 100% / ${sourceLen})`;
        lctDiv.style.width = `calc(100% / ${sourceLen})`;
        lctDiv.style.bottom = `calc(${lctStartY} * 100% / ${targetLen})`;
        lctDiv.style.height = `calc(${lctHeight} * 100% / ${targetLen})`;
    });
    
    while (chartUI.lctContainer.children.length > idealWindows.length) {
        chartUI.lctContainer.lastChild.remove();
    }
}
```

### 3. Interactive Markers/Tooltips

**Visual**: Circular markers with penalty/value labels
**Transition**: Smooth gliding between positions when data changes

#### CSS Requirements
```css
.marker-element {
    position: absolute; 
    width: 12px; 
    height: 12px; 
    border-radius: 50%;
    background-color: #8b5cf6; 
    border: 2px solid white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    z-index: 10;
    transition: left 0.4s var(--ease-out-cubic), 
                bottom 0.4s var(--ease-out-cubic), 
                transform 0.25s var(--ease-out-cubic), 
                background-color 0.25s var(--ease-out-cubic), 
                box-shadow 0.25s var(--ease-out-cubic);
    will-change: transform, left, bottom;
}

.marker-penalty-label {
    position: absolute; 
    top: 14px; 
    left: 50%; 
    transform: translateX(-50%);
    font-size: 0.7rem; 
    color: #dc2626; 
    font-weight: 600;
    background-color: rgba(255, 255, 255, 0.8); 
    padding: 1px 3px;
    border-radius: 2px;
    transition: opacity 0.2s ease;
}
```

#### JavaScript Implementation
```javascript
function updateMarkers(chartUI, markerData, sourceLen, targetLen) {
    const existingMarkers = Array.from(chartUI.markerContainer.children);
    
    markerData.forEach((marker, index) => {
        let markerDiv = existingMarkers[index];
        if (!markerDiv) {
            markerDiv = document.createElement('div');
            markerDiv.className = 'marker-element';
            const label = document.createElement('span');
            label.className = 'marker-penalty-label';
            markerDiv.appendChild(label);
            chartUI.markerContainer.appendChild(markerDiv);
        }
        
        // Update marker properties
        markerDiv.dataset.id = marker.id;
        const label = markerDiv.querySelector('.marker-penalty-label');
        
        if (marker.penalty > 0) {
            label.textContent = `-${marker.penalty}`;
            label.style.opacity = '1';
        } else {
            label.textContent = '';
            label.style.opacity = '0';
        }
        
        // Update position - CSS transitions handle animation
        markerDiv.style.left = `calc(${(marker.x + 0.5)} * 100% / ${sourceLen} - 6px)`;
        markerDiv.style.bottom = `calc(${(marker.y + 0.5)} * 100% / ${targetLen} - 6px)`;
    });
    
    while (chartUI.markerContainer.children.length > markerData.length) {
        chartUI.markerContainer.lastChild.remove();
    }
}
```

### 4. Animated Path Lines/Segments

**Visual**: SVG lines with drawing animations and color coding
**Transition**: Smooth path updates with staggered animations

#### CSS Requirements
```css
.path-segment-standard { stroke: #22c55e; stroke-width: 3px; }
.path-segment-lct-capped { stroke: #f59e0b; stroke-width: 3px; }
.path-segment-invalid { stroke: #ef4444; stroke-width: 3px; stroke-dasharray: 4 2; }

@keyframes draw-path {
    to { stroke-dashoffset: 0; }
}
```

#### JavaScript Implementation
```javascript
function updatePathSegments(chartUI, segments, chartWidth, chartHeight, numX, numY) {
    chartUI.actualPathSvg.innerHTML = '';
    
    segments.forEach((seg, index) => {
        if (!seg.start || !seg.end) return;
        
        const x1 = (seg.start.x + 0.5) * (chartWidth / numX);
        const y1 = chartHeight - ((seg.start.y + 0.5) * (chartHeight / numY));
        const x2 = (seg.end.x + 0.5) * (chartWidth / numX);
        const y2 = chartHeight - ((seg.end.y + 0.5) * (chartHeight / numY));
        
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        
        const length = Math.sqrt((x2-x1)**2 + (y2-y1)**2);
        line.setAttribute('class', 'path-segment-' + seg.calculation_method);
        line.style.strokeDasharray = length;
        line.style.strokeDashoffset = length;
        line.style.animation = `draw-path 0.6s var(--ease-out-cubic) ${index * 0.04}s forwards`;
        
        chartUI.actualPathSvg.appendChild(line);
    });
}
```

## Advanced Animation System

### Scenario Transition Animation

For smooth transitions when test cases change, implement position-based interpolation:

```javascript
function animateScenarioTransition(targetOrder, markerPositions, duration = 600) {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    
    const startPositions = JSON.parse(JSON.stringify(markerPositions));
    const targetPositions = calculateTargetPositions(targetOrder, startPositions);
    const startTime = performance.now();
    
    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }
    
    function tick(currentTime) {
        const elapsed = currentTime - startTime;
        const rawProgress = Math.min(elapsed / duration, 1);
        const progress = easeOutCubic(rawProgress);
        
        // Interpolate positions
        Object.keys(startPositions).forEach(chartType => {
            startPositions[chartType].forEach((start, i) => {
                const target = targetPositions[chartType][i];
                if (start && target) {
                    markerPositions[chartType][i].x = start.x + (target.x - start.x) * progress;
                    markerPositions[chartType][i].y = start.y + (target.y - start.y) * progress;
                }
            });
        });
        
        // Update visuals
        requestAnimationFrame(() => updateCharts(markerPositions));
        
        if (rawProgress < 1) {
            animationFrameId = requestAnimationFrame(tick);
        } else {
            animationFrameId = null;
            finalizeTransition(targetOrder);
        }
    }
    
    animationFrameId = requestAnimationFrame(tick);
}
```

## CSS Easing Variables

Define consistent easing curves for smooth animations:

```css
:root {
    --ease-out-cubic: cubic-bezier(0.215, 0.61, 0.355, 1);
    --ease-out-quad: cubic-bezier(0.25, 0.46, 0.45, 0.94);
    --ease-out-quart: cubic-bezier(0.165, 0.84, 0.44, 1);
}
```

## Implementation Checklist by Demo Type

### Basic Mapping Demo
- ✅ Mapping windows with smooth transitions
- ✅ Element reuse strategy
- ✅ CSS position transitions

### Marker + Mapping Demo  
- ✅ Mapping windows
- ✅ Interactive markers with smooth movement
- ✅ Penalty label transitions

### Full Grid Demo (like NAS)
- ✅ Mapping windows  
- ✅ LCT padding windows
- ✅ Interactive markers
- ✅ Path segments with draw animations
- ✅ Scenario transition animations

### Line/Path Demo
- ✅ Mapping windows
- ✅ Path markers
- ✅ Animated path drawing
- ✅ Smooth marker interpolation

## Performance Considerations

1. **Use `will-change`** for properties that will animate
2. **Limit concurrent animations** to prevent frame drops
3. **Use `requestAnimationFrame`** for smooth 60fps animations
4. **Cancel previous animations** before starting new ones
5. **Prefer `transform` over position** when possible for better performance

## Debugging Tips

1. **Add transition delays** temporarily to see animation order
2. **Use browser dev tools** to inspect computed styles during transitions
3. **Check for `innerHTML` clearing** that breaks element reuse
4. **Verify CSS transition properties** include all changing properties
5. **Use console logs** to track element reuse vs recreation

This documentation provides the foundation for implementing consistent, smooth grid effects across all visualization demos. 