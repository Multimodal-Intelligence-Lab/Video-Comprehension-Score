# Hero Section Implementation Guide for VCS Widget Demos

## Overview
This document provides exact specifications for implementing the teal-colored hero section with rounded top edges and square bottom edges across all VCS widget demos to maintain consistency with the main pages (playground.html, example.html, api.html).

## 1. HTML Structure Changes

### Before (Original Widget Structure)
```html
<div class="neumorphic-page-container">
    <div class="neumorphic-content">
        <!-- Main Container -->
        <main class="w-full max-w-8xl mx-auto">
            <!-- Header -->
            <div class="text-center mb-8 relative">
                <h1>Widget Title</h1>
                <!-- content -->
            </div>
            <!-- widget content -->
        </main>
    </div>
</div>
```

### After (With Hero Section)
```html
<div class="neumorphic-page-container">
    <div class="neumorphic-content">
        <article class="interactive-card overflow-hidden">
            <!-- Hero Section -->
            <section id="hero-section" class="py-6 px-4 sm:px-6 lg:px-8 relative">
                <div class="container mx-auto">
                    <!-- Start Tour Button - Top Right -->
                    <button id="start-tour-btn" class="hero-tour-btn absolute top-4 right-4 py-2 px-4">
                        <i class="fas fa-magic mr-2"></i>Start Tour
                    </button>
                    
                    <div class="text-left max-w-4xl">
                        <div class="flex items-center justify-start mb-4">
                            <div class="bg-teal-200 text-teal-800 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide mr-2">Interactive Demo</div>
                            <div class="bg-white bg-opacity-20 text-white px-3 py-1 rounded-full text-xs font-semibold">Core VCS Component</div>
                        </div>
                        <h1 class="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white mb-3 leading-tight">
                            [WIDGET_TITLE_HERE]
                        </h1>
                        <p class="text-teal-200 text-base sm:text-lg mb-6 max-w-3xl">
                            [WIDGET_DESCRIPTION_HERE]
                        </p>
                    </div>
                </div>
            </section>

            <!-- Main Container -->
            <main class="w-full max-w-8xl mx-auto p-6">
                <!-- widget content goes here -->
            </main>
        </article>
    </div>
</div>
```

## 2. CSS Styles to Add

### Hero Section Base Styles
```css
/* --- Hero Section Styles --- */
#hero-section {
    background: linear-gradient(45deg, #0d9488, #0f766e); 
    color: white; 
}
#hero-section h1 {
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2); 
}
```

### Hero Tour Button Styles (No White Glow)
```css
/* --- Hero Tour Button (No White Glow) --- */
.hero-tour-btn {
    background: linear-gradient(135deg, #0f766e 0%, #064e3b 100%);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 0.75rem;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.hero-tour-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
    transition: left 0.5s ease;
}

.hero-tour-btn:hover {
    background: linear-gradient(135deg, #14b8a6 0%, #0f766e 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    border-color: rgba(255, 255, 255, 0.3);
}

.hero-tour-btn:hover::before {
    left: 100%;
}

.hero-tour-btn:active {
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.5);
}
```

## 3. Typography & Color Specifications

### Title (H1) Specifications
- **Classes**: `text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white mb-3 leading-tight`
- **Font Family**: Inter (inherited from body)
- **Font Weight**: 800 (extrabold)
- **Color**: #ffffff (white)
- **Text Shadow**: `1px 1px 3px rgba(0,0,0,0.2)`
- **Responsive Sizes**: 
  - Mobile: text-3xl (30px)
  - Small: text-4xl (36px) 
  - Large: text-5xl (48px)

### Description (P) Specifications
- **Classes**: `text-teal-200 text-base sm:text-lg mb-6 max-w-3xl`
- **Color**: #99f6e4 (teal-200)
- **Font Size**: 
  - Mobile: text-base (16px)
  - Small+: text-lg (18px)
- **Max Width**: 48rem (768px)

### Badge Specifications
- **First Badge**: `bg-teal-200 text-teal-800 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide mr-2`
  - Background: #99f6e4 (teal-200)
  - Text: #115e59 (teal-800)
  - Text: "Interactive Demo"
- **Second Badge**: `bg-white bg-opacity-20 text-white px-3 py-1 rounded-full text-xs font-semibold`
  - Background: rgba(255, 255, 255, 0.2)
  - Text: #ffffff (white)
  - Text: "Core VCS Component"

## 4. Background & Container Specifications

### Hero Background
- **Background**: `linear-gradient(45deg, #0d9488, #0f766e)`
- **Colors**: 
  - Start: #0d9488 (teal-600)
  - End: #0f766e (teal-700)
- **Angle**: 45 degrees

### Container Structure
- **Outer Container**: `neumorphic-page-container` (maintains rounded edges)
- **Content Wrapper**: `neumorphic-content`
- **Article Wrapper**: `interactive-card overflow-hidden` (creates rounded top edges, square bottom)
- **Section**: `py-6 px-4 sm:px-6 lg:px-8 relative` (compact height padding)

## 5. Button Positioning & Behavior

### Start Tour Button
- **Position**: `absolute top-4 right-4`
- **Size**: `py-2 px-4` (compact padding)
- **Icon**: FontAwesome `fas fa-magic`
- **Text**: "Start Tour"
- **Z-index**: Inherits from positioning context

### Button States
- **Default**: Dark teal gradient with subtle shadow
- **Hover**: Lighter teal, lifts up 2px, stronger shadow
- **Active**: Lifts up 1px, stronger shadow
- **Focus**: Same as hover

## 6. JavaScript Event Handling

### Required Event Listener
```javascript
// In window.addEventListener('load', () => { ... })
document.getElementById('start-tour-btn').addEventListener('click', () => tourManager.start());
```

### Prerequisites
- Tour manager must be defined and functional
- Button ID must match: `start-tour-btn`

## 7. Responsive Behavior

### Desktop (lg+)
- Title: text-5xl (48px)
- Description: text-lg (18px)
- Button: Top-right corner, fixed position

### Tablet (sm to lg)
- Title: text-4xl (36px)
- Description: text-lg (18px)
- Button: Top-right corner, adjusted padding

### Mobile (< sm)
- Title: text-3xl (30px)
- Description: text-base (16px)
- Button: Top-right corner, compact size

## 8. Implementation Checklist

### HTML Changes
- [ ] Wrap content in `<article class="interactive-card overflow-hidden">`
- [ ] Add hero section with exact class structure
- [ ] Update title with responsive classes
- [ ] Add description with proper styling
- [ ] Include both badges with correct colors
- [ ] Add Start Tour button in top-right corner
- [ ] Close article tag after main content

### CSS Changes
- [ ] Add `#hero-section` styles with gradient background
- [ ] Add `#hero-section h1` text shadow
- [ ] Add complete `.hero-tour-btn` style block
- [ ] Include all hover and active states
- [ ] Ensure backdrop-filter support

### JavaScript Changes
- [ ] Add event listener for `start-tour-btn`
- [ ] Ensure tour manager is available
- [ ] Test button functionality

### Content Customization
- [ ] Replace `[WIDGET_TITLE_HERE]` with widget-specific title
- [ ] Replace `[WIDGET_DESCRIPTION_HERE]` with widget-specific description
- [ ] Ensure description is concise and under 768px width

## 9. Widget-Specific Customizations

### Title Examples
- "Mapping Window Visualization"
- "Global Alignment Score (GAS)"
- "Local Alignment Score (LAS)"
- "Best Matching Algorithm"
- "Distance-based Narrative Alignment Score (NAS-D) Visualization"
- "Line-based Narrative Alignment Score (NAS-L) Flow Analysis"

### Description Pattern
Follow this pattern: "Explore [CONCEPT] in the Video Comprehension Score framework."

### Common Descriptions
- "Explore how texts of different lengths are aligned using precision and recall mapping windows in the Video Comprehension Score framework."
- "Understand global thematic similarity measurement and its role in comprehensive text evaluation."
- "Dive into fine-grained semantic quality assessment of matched text chunks."

## 10. Testing Verification

### Visual Checks
- [ ] Teal gradient background displays correctly
- [ ] Rounded top edges, square bottom edges visible
- [ ] Button positioned in top-right corner
- [ ] Typography matches main pages exactly
- [ ] Responsive behavior works on all screen sizes

### Functional Checks
- [ ] Start Tour button triggers tour functionality
- [ ] Button hover effects work smoothly
- [ ] No console errors on page load
- [ ] Hero section height is appropriate (not too tall)

### Cross-Widget Consistency
- [ ] Hero section height consistent across all widgets
- [ ] Button styling identical across all widgets
- [ ] Typography specifications match exactly
- [ ] Color values are identical

## 11. Color Reference

### Exact Color Values
- **Hero Background Start**: #0d9488
- **Hero Background End**: #0f766e
- **Title Text**: #ffffff
- **Description Text**: #99f6e4
- **Badge 1 Background**: #99f6e4
- **Badge 1 Text**: #115e59
- **Badge 2 Background**: rgba(255, 255, 255, 0.2)
- **Badge 2 Text**: #ffffff
- **Button Background Start**: #0f766e
- **Button Background End**: #064e3b
- **Button Hover Start**: #14b8a6
- **Button Hover End**: #0f766e
- **Button Border**: rgba(255, 255, 255, 0.2)
- **Button Shadow**: rgba(0, 0, 0, 0.3)

This documentation provides complete specifications for implementing the hero section across all VCS widget demos with pixel-perfect consistency. 