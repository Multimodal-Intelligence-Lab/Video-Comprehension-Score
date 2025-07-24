# Neumorphic Style Update Documentation

This document outlines all the styling changes made to transform the best-match demo into a neumorphic design while preserving the original color palette and functionality. Use this as a reference to apply similar transformations to other widget pages.

## 1. Core Theme Variables

### CSS Variables Added
```css
:root {
    --bg-color: #f8fafc; /* Keep original light background */
    --light-shadow: rgba(255, 255, 255, 1);
    --dark-shadow: rgba(148, 163, 184, 0.3); /* Subtle slate shadows */
    --ease-out-quad: cubic-bezier(0.25, 0.46, 0.45, 0.94);
    --ease-out-cubic: cubic-bezier(0.215, 0.610, 0.355, 1.000);
}
```

### Body Updates
```css
body {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-color);
    transition: background 1.5s cubic-bezier(0.25, 0.1, 0.25, 1);
}
```

## 2. Page Container Structure

### HTML Structure Added
```html
<!-- Wrap entire content in neumorphic container -->
<div class="neumorphic-page-container">
    <div class="neumorphic-content">
        <!-- Main Container -->
        <main class="w-full max-w-7xl mx-auto">
            <!-- All existing content -->
        </main>
    </div>
</div>
```

### Container Styling
```css
.neumorphic-page-container {
    background: var(--bg-color);
    border-radius: 25px;
    box-shadow: 
        12px 12px 24px var(--dark-shadow), 
        -12px -12px 24px var(--light-shadow);
    margin: 1rem;
    padding: 2rem;
    min-height: calc(100vh - 2rem);
    position: relative;
    overflow: hidden;
}

.neumorphic-content {
    position: relative;
    z-index: 1;
}
```

## 3. Card Components

### Interactive Cards (Base)
```css
.interactive-card {
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    box-shadow: 8px 8px 16px var(--dark-shadow), -8px -8px 16px var(--light-shadow);
    transition: transform 0.35s var(--ease-out-cubic),
                box-shadow 0.35s var(--ease-out-cubic),
                border-color 0.35s var(--ease-out-cubic);
    will-change: transform, box-shadow;
}

.interactive-card:hover {
    box-shadow: 12px 12px 24px var(--dark-shadow), -12px -12px 24px var(--light-shadow);
    transform: translateY(-5px);
    border-color: rgba(255, 255, 255, 0.5);
}
```

### Demo Cards (Teal Theme)
```css
.demo-card { 
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.75) 0%, rgba(248, 250, 252, 0.65) 100%); 
    border-color: rgba(226, 232, 240, 0.8);
    border-left: 3px solid #0d9488;
    border-radius: 20px;
    box-shadow: 8px 8px 16px var(--dark-shadow), -8px -8px 16px var(--light-shadow), 0 2px 8px rgba(0, 0, 0, 0.06), 0 1px 3px rgba(0, 0, 0, 0.04);
}
```

### Intro Cards (Amber Theme)
```css
.intro-card { 
    background: linear-gradient(135deg, rgba(254, 252, 232, 0.6) 0%, rgba(254, 243, 199, 0.5) 100%); 
    border-color: rgba(251, 191, 36, 0.5);
    border-radius: 20px;
    box-shadow: 8px 8px 16px var(--dark-shadow), -8px -8px 16px var(--light-shadow);
}
```

## 4. Chart and Grid Components

### Chart Grid Background
```css
.chart-grid-background {
    background-color: #fafbfc;
    border-radius: 15px;
    box-shadow: inset 5px 5px 10px var(--dark-shadow), inset -5px -5px 10px var(--light-shadow), inset 0 1px 3px rgba(0, 0, 0, 0.05);
}
```

### Grid Lines (if applicable)
```css
.grid-line {
    position: absolute;
    background-color: rgba(148, 163, 184, 0.3);
    z-index: 0;
    pointer-events: none;
}
```

## 5. Input Elements

### Neumorphic Inputs
```css
.neumorphic-input {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    box-shadow: inset 4px 4px 8px var(--dark-shadow), inset -4px -4px 8px var(--light-shadow);
    transition: box-shadow 0.3s var(--ease-out-quad), border-color 0.3s var(--ease-out-quad);
    color: #374151;
}

.neumorphic-input:focus {
    outline: none;
    box-shadow: inset 6px 6px 12px var(--dark-shadow), inset -6px -6px 12px var(--light-shadow);
    border-color: #0d9488;
}
```

### HTML Input Updates
Replace existing input classes with:
```html
<input type="number" class="w-full p-3 neumorphic-input">
```

## 6. Button Components

### Neumorphic Buttons
```css
.neumorphic-btn {
    border-radius: 12px;
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
    color: white;
    font-weight: 600;
    box-shadow: 5px 5px 10px var(--dark-shadow), -5px -5px 10px var(--light-shadow);
    transition: all 0.2s var(--ease-out-quad);
    border: none;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.neumorphic-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s ease;
}

.neumorphic-btn:hover {
    box-shadow: 6px 6px 12px var(--dark-shadow), -6px -6px 12px var(--light-shadow);
    transform: translateY(-1px) scale(1.02);
}

.neumorphic-btn:hover::before {
    left: 100%;
}

.neumorphic-btn:active {
    box-shadow: inset 4px 4px 8px var(--dark-shadow), inset -4px -4px 8px var(--light-shadow);
    transform: translateY(1px);
}

.neumorphic-btn.primary {
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
}
```

### HTML Button Updates
Replace existing button classes with:
```html
<!-- Regular buttons -->
<button class="neumorphic-btn w-full py-3 text-sm">Button Text</button>

<!-- Primary buttons -->
<button class="neumorphic-btn primary py-2 px-4">Primary Button</button>
```

## 7. Tour and Modal Components

### Tour Callout
```css
#tour-callout { 
    position: absolute; 
    background: rgba(255, 255, 255, 0.95); 
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: #334155; 
    padding: 1.25rem; 
    border-radius: 20px; 
    box-shadow: 8px 8px 16px var(--dark-shadow), -8px -8px 16px var(--light-shadow);
    z-index: 10000; 
    max-width: 320px; 
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: opacity 0.3s var(--ease-out-cubic), transform 0.3s var(--ease-out-cubic); 
    transform: translateY(10px); 
    opacity: 0; 
}

#tour-callout button { 
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%); 
    color: white; 
    padding: 0.5rem 1rem; 
    border: none; 
    border-radius: 12px; 
    cursor: pointer; 
    font-weight: 600;
    box-shadow: 4px 4px 8px var(--dark-shadow), -4px -4px 8px var(--light-shadow);
    transition: all 0.2s var(--ease-out-quad);
}

#tour-callout button:hover { 
    box-shadow: 6px 6px 12px var(--dark-shadow), -6px -6px 12px var(--light-shadow);
    transform: translateY(-1px) scale(1.02); 
}
```

### Testing Modal
```css
.testing-modal-content {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 25px;
    padding: 2.5rem;
    max-width: 90vw;
    max-height: 85vh;
    width: 700px;
    box-shadow: 
        12px 12px 24px var(--dark-shadow), 
        -12px -12px 24px var(--light-shadow),
        0 8px 32px rgba(0, 0, 0, 0.1);
    transform: translateY(30px) scale(0.9);
    transition: all 0.5s var(--ease-out-cubic);
    overflow-y: auto;
    box-sizing: border-box;
    position: relative;
}

.testing-modal-close {
    background: rgba(255, 255, 255, 0.9);
    border: none;
    font-size: 1.25rem;
    cursor: pointer;
    color: #64748b;
    padding: 0.75rem;
    border-radius: 50%;
    box-shadow: 4px 4px 8px var(--dark-shadow), -4px -4px 8px var(--light-shadow);
    transition: all 0.3s var(--ease-out-cubic);
    width: 3rem;
    height: 3rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.testing-modal-close:hover {
    color: #0d9488;
    transform: rotate(90deg) scale(1.1);
    box-shadow: 6px 6px 12px var(--dark-shadow), -6px -6px 12px var(--light-shadow);
}
```

### Test Categories and Items
```css
.test-category {
    margin-bottom: 2rem;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    box-shadow: 6px 6px 12px var(--dark-shadow), -6px -6px 12px var(--light-shadow);
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.test-category:hover {
    transform: translateY(-2px);
    box-shadow: 8px 8px 16px var(--dark-shadow), -8px -8px 16px var(--light-shadow);
}

.test-item {
    margin-bottom: 1.5rem;
    padding: 1.5rem;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.4);
    box-shadow: 4px 4px 8px var(--dark-shadow), -4px -4px 8px var(--light-shadow);
    transition: all 0.2s var(--ease-out-quad);
}

.test-item:hover {
    transform: translateY(-1px);
    box-shadow: 6px 6px 12px var(--dark-shadow), -6px -6px 12px var(--light-shadow);
}
```

## 8. Implementation Steps

### Step 1: Add CSS Variables
Add the root variables at the top of your CSS.

### Step 2: Wrap Content
Wrap your main content in the neumorphic page container structure.

### Step 3: Update Card Classes
Apply the new card styles while preserving original color gradients.

### Step 4: Update Interactive Elements
Replace input and button classes with neumorphic versions.

### Step 5: Update Modals and Tours
Apply neumorphic styling to overlay elements.

### Step 6: Test and Adjust
Ensure all hover states and interactions work smoothly.

## 9. Key Principles

### Color Preservation
- **Always maintain** original color schemes and gradients
- **Only enhance** with neumorphic shadows and structure
- **Keep** brand colors (teal, amber, etc.) intact

### Shadow System
- **Light shadow**: `rgba(255, 255, 255, 1)` for highlights
- **Dark shadow**: `rgba(148, 163, 184, 0.3)` for depth
- **Consistent scaling**: 4px/8px for small, 8px/16px for medium, 12px/24px for large

### Border Radius
- **Small elements**: 10-12px
- **Cards**: 15-20px
- **Containers**: 20-25px
- **Buttons**: 12px
- **Circular**: 50% for close buttons

### Transitions
- **Standard**: 0.3s var(--ease-out-quad)
- **Transform**: 0.35s var(--ease-out-cubic)
- **Quick interactions**: 0.2s var(--ease-out-quad)

## 10. Browser Compatibility Notes

- Use `-webkit-backdrop-filter` for Safari support
- Test shadow rendering across different browsers
- Ensure fallbacks for older browsers if needed
- Maintain accessibility contrast ratios

---

**Note**: This documentation preserves all original functionality while adding neumorphic visual enhancements. The key is to enhance, not replace, the existing design language. 