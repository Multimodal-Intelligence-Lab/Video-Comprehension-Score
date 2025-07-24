# Algorithm Section Style Guide

## Overview
This document defines the standard styling, structure, and implementation approach for algorithm sections across all VCS widget demos. It ensures consistency in presentation, tone, and user experience.

## Table of Contents
1. [Visual Design](#visual-design)
2. [CSS Requirements](#css-requirements)
3. [HTML Structure](#html-structure)
4. [Content Guidelines](#content-guidelines)
5. [Mathematical Notation](#mathematical-notation)
6. [Writing Style](#writing-style)
7. [Implementation Checklist](#implementation-checklist)

---

## Visual Design

### Container Structure
- **Main container**: White background with amber/orange left border (4px width)
- **Algorithm overview**: Brief academic-style introduction paragraph
- **Individual steps**: Orange-themed blocks with prerequisite-style formatting
- **Mathematical equations**: Blue-themed containers matching prerequisite blocks

### Color Palette
- **Step containers**: `bg-orange-50` background, `border-orange-200` border
- **Step titles**: `text-orange-800` for headers, `text-orange-700` for body text
- **Icons**: `text-orange-500` for FontAwesome icons
- **Math containers**: `#dbeafe` background, `#93c5fd` border (blue theme)
- **Math equations**: `#1e40af` color (blue-700)
- **Conclusion block**: `bg-blue-50` background, `border-blue-200` border
- **Conclusion text**: `text-blue-800` for headers, `text-blue-700` for body text
- **Conclusion icon**: `text-blue-500` for play-circle icon

### Typography
- **Step titles**: `text-xs font-medium` 
- **Body text**: `text-xs` with `text-orange-700` color
- **Math text**: `text-xs` within blue containers
- **Line spacing**: `mb-1` for titles, `mb-2` for paragraphs, `mt-2` for follow-up text

### Layout & Spacing
- **Step spacing**: `mt-3` between steps
- **Container padding**: `p-3` for step containers
- **Math container**: `padding: 0.75rem`, `margin-top: 0.75rem`
- **Border radius**: `rounded-xl` for steps, `border-radius: 0.75rem` for math

---

## CSS Requirements

### KaTeX Integration
```html
<!-- Add to <head> section -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    });
</script>
```

### Required CSS Classes
```css
/* Math container styling */
.math-container {
    background: #dbeafe;
    border: 1px solid #93c5fd;
    border-radius: 0.75rem;
    padding: 0.75rem;
    margin-top: 0.75rem;
}

/* KaTeX color overrides */
.katex {
    color: #1e40af !important;
}

.katex .mord,
.katex .mop,
.katex .mbin,
.katex .mrel,
.katex .mopen,
.katex .mclose,
.katex .mpunct {
    color: #1e40af !important;
}

.katex .mtext {
    color: #1e3a8a !important;
}
```

---

## HTML Structure

### Main Container Template
```html
<details id="tour-step-math" class="mb-4">
    <summary class="font-semibold text-amber-700 mb-3 flex items-center">
        <i class="fas fa-chevron-right summary-icon"></i>
        [Algorithm Name] Algorithm
    </summary>
    <div class="mt-4 space-y-4">
        <!-- Main Algorithm Overview -->
        <div class="bg-white bg-opacity-70 p-4 rounded-lg border-l-4 border-amber-500">
            <p class="text-sm text-slate-700 mb-3">
                <strong>Algorithm Overview:</strong> [Brief academic description]
            </p>
            
            <!-- Individual Steps -->
            [Step containers go here]
            
            <!-- Algorithm Conclusion -->
            <div class="bg-blue-50 border border-blue-200 rounded-xl p-3 mt-3">
                <div class="flex items-start">
                    <i class="fas fa-play-circle text-blue-500 mr-2 mt-0.5"></i>
                    <div>
                        <p class="text-xs text-blue-800 font-medium mb-1">🎯 Interactive Demo</p>
                        <p class="text-xs text-blue-700 mb-2">[Brief transition from algorithm to demo]</p>
                        <p class="text-xs text-blue-700">[Encouragement to experiment with demo features]</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</details>
```

### Step Container Template
```html
<div class="bg-orange-50 border border-orange-200 rounded-xl p-3 mt-3">
    <div class="flex items-start">
        <i class="fas fa-[icon-name] text-orange-500 mr-2 mt-0.5"></i>
        <div>
            <p class="text-xs text-orange-800 font-medium mb-1">Step X: [Step Title]</p>
            <p class="text-xs text-orange-700 mb-2">[Step description]</p>
            [Optional: Additional context paragraph]
            
            <!-- Math Container -->
            <div class="math-container">
                <p class="text-xs">$[LaTeX equation]$</p>
                [Additional equations]
            </div>
            
            <p class="text-xs text-orange-700 mt-2">[Follow-up explanation]</p>
        </div>
    </div>
</div>
```

---

## Content Guidelines

### Algorithm Overview
- **Length**: 1-2 sentences maximum
- **Tone**: Academic but accessible
- **Focus**: High-level purpose and methodology
- **Format**: Start with "The [algorithm name] algorithm [verb describing main action]..."

### Step Descriptions
- **Structure**: Title + 2-3 explanation paragraphs + mathematical formulation + follow-up
- **Progression**: Each step builds logically on previous steps
- **Clarity**: Explain the "why" not just the "what"
- **Context**: Connect to broader algorithmic purpose

### Algorithm Conclusion
- **Purpose**: Bridge algorithm explanation to interactive demo
- **Placement**: After all algorithm steps, before closing details tag
- **Theme**: Blue color scheme to differentiate from orange steps
- **Content**: Encourage experimentation with demo features
- **Length**: 1-2 sentences maximum per paragraph
- **Icon**: Use `fa-play-circle` to suggest interactivity

### Mathematical Integration
- **Placement**: After conceptual explanation, before technical details
- **Presentation**: Clean LaTeX in left-aligned blue containers
- **Alignment**: Left-justified equations with proper baseline alignment
- **Explanation**: Always follow equations with interpretation

---

## Mathematical Notation

### Standard Variables
- **Text inputs**: $T_{ref}$, $T_{gen}$ 
- **Chunk sets**: $C_{ref}$, $C_{gen}$
- **Chunk counts**: $N_{ref}$, $N_{gen}$
- **Length variables**: $N_{long}$, $N_{short}$, $N_{longer}$, $N_{shorter}$
- **Window functions**: $W_{direct}(i)$, $W_{reverse}(j)$
- **Final windows**: $MW_{prec}(i)$, $MW_{rec}(j)$

### Formatting Rules
- **Inline math**: Use `$...$` for variables within text
- **Display math**: Use `$...$` within `<p class="text-xs">` tags in left-aligned math containers
- **Sets**: Use `\{...\}` for set notation
- **Intervals**: Use `[a, b)` for half-open intervals
- **Functions**: Use subscripts for clarity: `\text{function}_{subscript}`

### KaTeX-Specific Syntax Requirements
- **Variable names with underscores**: Always escape underscores with `\_`
  - Correct: `\text{max\_val}`, `\text{context\_cutoff}`
  - Incorrect: `\text{max_val}`, `\text{context_cutoff}`
- **Arg operators**: Use `\underset{}{\arg\min}` and `\underset{}{\arg\max}` instead of subscripts
  - Correct: `\underset{i \in C}{\arg\min} \; d_i`
  - Incorrect: `\arg\min_{i \in C} d_i`
- **Array indexing**: Use subscripts instead of brackets
  - Correct: `\text{similarity\_array}_i`
  - Incorrect: `\text{similarity_array}[i]`
- **Complex cases**: Avoid multiline `\begin{cases}` - use separate equations instead
- **Spacing**: Use `\;` for proper spacing between operators and operands

### Mathematical Concepts
- **Floor function**: `\lfloor x \rfloor`
- **Ceiling function**: `\lceil x \rceil`
- **Set membership**: `\in`, `\notin`
- **Logical operators**: `\land`, `\lor`, `\neg`
- **Conditional expressions**: Use `\begin{cases}...\end{cases}` for piecewise functions

---

## Writing Style

### Tone
- **Academic**: Precise, formal language
- **Accessible**: Avoid unnecessary jargon
- **Explanatory**: Focus on understanding over implementation
- **Progressive**: Build complexity gradually

### Voice
- **Active voice**: "The algorithm computes..." not "The ratio is computed..."
- **Present tense**: Describe what happens, not what will happen
- **Authoritative**: Confident explanations without hedging
- **Inclusive**: Use "we" for collaborative tone

### Sentence Structure
- **Concise**: Maximum 20-25 words per sentence
- **Parallel**: Consistent structure across similar concepts
- **Logical flow**: Each sentence connects to the next
- **Transitions**: Use connective phrases between concepts

### Technical Language
- **Precision**: Use exact mathematical terms
- **Consistency**: Same terms for same concepts throughout
- **Definition**: Introduce technical terms before using them
- **Abbreviations**: Define acronyms on first use

---

## Implementation Checklist

### Before Starting
- [ ] MathJax scripts added to `<head>`
- [ ] CSS classes defined for math containers
- [ ] MathJax color overrides implemented
- [ ] FontAwesome icons available

### Content Development
- [ ] Algorithm overview written (1-2 sentences)
- [ ] Step titles follow naming convention
- [ ] Each step has conceptual explanation
- [ ] Mathematical notation is consistent
- [ ] Follow-up explanations provided
- [ ] Algorithm conclusion block added
- [ ] Conclusion encourages demo interaction

### Visual Implementation
- [ ] Main container has amber left border
- [ ] Step containers use orange theme
- [ ] Math containers use blue theme
- [ ] Conclusion block uses blue theme
- [ ] Icons are semantically appropriate
- [ ] Spacing follows specification

### Technical Verification
- [ ] KaTeX renders correctly
- [ ] All LaTeX equations compile
- [ ] Equations display properly in Safari and Chrome
- [ ] Equations are left-justified (no text-center class)
- [ ] All underscores in variable names are escaped (`\_`)
- [ ] Arg operators use `\underset{}{\arg\min}` syntax
- [ ] No array bracket notation (`[i]`) used
- [ ] Complex cases split into separate equations
- [ ] Colors match specification
- [ ] Responsive design maintained
- [ ] Accessibility standards met

### Quality Assurance
- [ ] Content flows logically
- [ ] Mathematical accuracy verified
- [ ] Consistent tone throughout
- [ ] No spelling/grammar errors
- [ ] Visual hierarchy clear

---

## Icon Selection Guide

### Recommended Icons for Common Steps
- **Text processing**: `fa-scissors`, `fa-file-text`
- **Calculations**: `fa-calculator`, `fa-function`
- **Comparisons**: `fa-balance-scale`, `fa-arrows-alt-h`
- **Transformations**: `fa-arrow-right`, `fa-arrow-left`
- **Assignments**: `fa-tags`, `fa-assign`
- **Windows/Regions**: `fa-window-maximize`, `fa-border-all`
- **Interpolation**: `fa-chart-line`, `fa-arrows-alt`
- **Construction**: `fa-tools`, `fa-cogs`

### Icon Color Standard
- All icons should use `text-orange-500` class
- Icons should be positioned with `mr-2 mt-0.5` for proper alignment
- Use semantic icons that relate to the step's function

---

## Notes for Future Implementation

### Consistency Requirements
- All algorithm sections across widgets must follow this exact specification
- Any deviations require updating this documentation first
- Visual design should be identical across all implementations

### Extensibility
- This template can accommodate 3-8 algorithm steps
- Complex algorithms may require sub-steps (use nested structure)
- Mathematical complexity can vary while maintaining visual consistency

### Maintenance
- Update this document when making any changes to the standard
- Test changes across multiple widgets before finalizing
- Keep mathematical notation consistent across all VCS demos
- Ensure KaTeX integration is properly implemented for consistent cross-browser rendering
- Test equation rendering in both Chrome and Safari to ensure cross-browser compatibility
- Verify all KaTeX syntax requirements are followed when adding new equations
- Use `throwOnError: false` in KaTeX configuration to prevent page crashes

---

## Troubleshooting Common KaTeX Issues

### Equations Not Rendering
**Problem**: Math equations appear as raw LaTeX text instead of rendered math.
**Solutions**:
1. Check that KaTeX scripts are loaded correctly
2. Verify `renderMathInElement` is called after DOM content loads
3. Ensure delimiters are correctly configured (`$...$` for inline, `$$...$$` for display)
4. Add `throwOnError: false` to prevent crashes on syntax errors

### Underscore-Related Errors
**Problem**: Variables like `max_val` cause rendering failures.
**Solution**: Always escape underscores: `\text{max\_val}` instead of `\text{max_val}`

### Arg Operator Issues
**Problem**: `\arg\min_{subscript}` or `\arg\max_{subscript}` don't render properly.
**Solution**: Use `\underset{subscript}{\arg\min}` syntax instead

### Array Indexing Problems
**Problem**: Bracket notation like `array[i]` causes errors.
**Solution**: Use subscript notation: `\text{array}_i`

### Complex Cases Not Working
**Problem**: Multiline `\begin{cases}` statements fail to render.
**Solution**: Split complex cases into separate equations with conditional text

### Safari-Specific Rendering Issues
**Problem**: Equations appear "wavy" or misaligned in Safari.
**Solution**: KaTeX generally resolves Safari issues better than MathJax - ensure proper syntax

---

*This document serves as the single source of truth for algorithm section implementation across all VCS widget demonstrations. Any changes to styling, structure, or content approach must be reflected in this documentation to maintain consistency.*