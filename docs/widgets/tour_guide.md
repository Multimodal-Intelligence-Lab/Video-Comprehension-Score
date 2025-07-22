# Tour Guide Specification

## Philosophy

Tour guides are **component-focused signposts**, not educational content. They explain *what* each UI element does and *how* to interact with it, without repeating theoretical content that belongs in Testing Guides or Introduction sections.

## Core Guidelines

### ✅ What Tour Messages SHOULD Have

1. **Component Purpose** - Clear explanation of what this UI element does
2. **Interaction Instructions** - How to use/interact with this specific component
3. **Visual Guidance** - What to look for or expect when using it
4. **Navigation Hints** - Where to find related information (e.g., "Check Testing Guide for details")
5. **Practical Actions** - Specific things to try or click

### ❌ What Tour Messages SHOULD NOT Have

1. **Theoretical Explanations** - Mathematical concepts belong in Testing Guide/Algorithm sections
2. **Repeated Information** - Don't explain the same concept across multiple tour steps
3. **Background Context** - Detailed algorithm theory should be in dedicated sections
4. **Redundant Descriptions** - Avoid repeating what's already visible in the UI
5. **Technical Deep-dives** - Keep focused on UI interaction, not algorithm internals

## Message Structure Template

```
[Component Name/Purpose] + [How to interact] + [What you'll see/get] + [Where to learn more if needed]
```

## Length Guidelines

- **1-3 sentences maximum** per tour step
- **Focus on actionable information**
- **Direct and conversational tone**
- **Avoid technical jargon** unless essential for interaction

## Standard Tour Message Order

### Step 1: Welcome (REQUIRED)
- **Always starts with "Welcome to [Widget Name]!"**
- Brief 1-sentence problem statement
- Prerequisites/dependencies reference if applicable
- Position: `'bottom'`

### Step 2: Algorithm/Core Concept (Dropdown/Expandable)
- Expandable algorithm or main concept section
- Focus on interaction ("Click to expand")
- **DO NOT mention "live updates"** - algorithm sections show static theory/math
- Briefly describe what's inside the dropdown (e.g., "mathematical theory", "algorithm steps")
- Position: `'bottom'`

### Step 3: Main Visualization
- Primary interactive chart/visualization
- Drag/interaction instructions
- What visual changes to expect
- Position: `'top'` (usually)

### Step 4: Controls
- Input controls, sliders, dimension settings
- How they connect to visualization
- Specific scenarios to try
- Position: `'left'` or `'right'`

### Step 5: Secondary Features
- Additional interactive elements (penalty bars, overlays, etc.)
- Hover effects or special interactions
- Visual feedback explanations
- Position: varies based on layout

### Step 6: Presets/Scenarios
- Preset buttons or scenario selection
- Key comparisons to try
- Reference to Testing Guide for details
- Position: `'left'` or `'right'`

### Step 7: Statistics/Results
- Live statistics display
- What each metric means in practical terms
- Real-time update behavior
- Position: `'left'` (usually)

## Examples

### Good Examples ✅

```javascript
// Welcome message
{ selector: '#tour-step-intro', text: 'Welcome to Window Regularizer! This demo solves the problem where unbalanced text lengths create unfairly large alignment windows. Visit Distance-NAS and Line-NAS demos first to understand the foundational concepts.', position: 'bottom' }

// Algorithm dropdown - shows static theory/math
{ selector: '#tour-step-algo', text: 'Click to expand the algorithm breakdown. This shows the 3-step mathematical theory behind window regularization penalties.', position: 'bottom' }

// Interactive visualization
{ selector: '#tour-step-chart', text: 'This chart shows your mapping windows as blue rectangles. Drag the edges to resize the grid and watch the coverage percentage change. The coverage ratio determines when penalties kick in.', position: 'top' }
```

### Bad Examples ❌

```javascript
// Too theoretical
{ selector: '#tour-step-intro', text: 'Welcome to the sophisticated regularization technique that addresses low dimensionality problems through area-based coverage analysis using mathematical foundations involving non-linear penalty computations...', position: 'bottom' }

// Incorrect about live updates (algorithm dropdowns show static content)
{ selector: '#tour-step-algo', text: 'Click to expand the 3-step algorithm breakdown. All calculations update live as you interact with the demo. Check here to understand the math behind the penalty calculations.', position: 'bottom' }

// Too repetitive  
{ selector: '#tour-step-chart', text: 'This visualization shows coverage ratios which are fundamental metrics that calculate area percentages for regularization penalties that determine score adjustments...', position: 'top' }
```

## Implementation Checklist

When implementing tour guides for new widgets:

- [ ] Start with "Welcome to [Widget Name]!"
- [ ] Each message is 1-3 sentences maximum
- [ ] Focus on UI interaction, not theory
- [ ] Include specific actions to try
- [ ] Reference Testing Guide for detailed explanations
- [ ] Use appropriate positioning for each step
- [ ] Test tour flow for logical progression
- [ ] Ensure no repeated information between steps

## Common Positions

- **`'bottom'`** - Welcome messages, algorithm sections (expandable content)
- **`'top'`** - Main charts, visualizations (usually at top of screen)
- **`'left'`** - Control panels, preset buttons (usually on left side)
- **`'right'`** - Secondary controls, statistics (usually on right side)

## Tone & Voice

- **Conversational** - "Try these preset scenarios"
- **Direct** - "Click to expand", "Drag the edges"
- **Helpful** - "Check Testing Guide for details"
- **Encouraging** - "Watch the coverage percentage change"

## Quick Reference for Future Prompts

**To implement this specification, use this prompt:**

"Apply the tour guide specification from `/docs/widgets/tour_guide.md` to create/update tour messages for [Widget Name]. Follow the standard order, message structure, and guidelines exactly as specified in the document."