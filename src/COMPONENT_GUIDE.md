# Component Structure Overview

## 📐 Architecture

The landing page is built with a **modular component-based architecture** where each section is a separate, reusable React component with its own styling.

## 🗂️ File Organization

### Root Level
- `index.html` - HTML entry point
- `package.json` - Dependencies and scripts
- `vite.config.js` - Build configuration
- `.gitignore` - Git ignore rules
- `README.md` - Documentation

### src/
Main source directory containing all application code

#### src/App.jsx
**Main application component** that imports and renders all sections in order:
- Header
- Hero (with ArpeggiatorDemo)
- Features
- Experience
- Founder
- Trusted
- Testimonial
- Footer

#### src/index.jsx
**Entry point** that renders the App component to the DOM

#### src/styles/
Global styles and CSS variables

**GlobalStyles.css**
- CSS custom properties (variables) for colors, fonts, spacing
- Global reset styles
- Animation keyframes
- Responsive container styles
- Base body styling with gradient background

#### src/components/
All React components with co-located CSS files

---

## 🧩 Components Detail

### 1. Header (Header.jsx + Header.css)
**Purpose:** Top navigation bar
**Contains:**
- Logo with music icon
- Navigation links (Builder, Profile)
- Responsive behavior (hides nav on mobile)

### 2. Hero (Hero.jsx + Hero.css)
**Purpose:** Main headline and call-to-action
**Contains:**
- Large heading with "energetic" in orange
- ArpeggiatorDemo component
- "Start Building" CTA button

### 3. ArpeggiatorDemo (ArpeggiatorDemo.jsx + ArpeggiatorDemo.css)
**Purpose:** Interactive music interface demo
**Contains:**
- 12-note grid (C through B) with click toggles
- Input fields for pattern customization
- Generate button
- Visualization area
- Export/Download/Refine action buttons
**State:** Manages active notes array (default: C major chord)

### 4. Features (Features.jsx + Features.css)
**Purpose:** Showcase main features in a grid
**Contains:**
- Section title
- Grid of FeatureCard components
- Passes props to FeatureCard for customization

### 5. FeatureCard (FeatureCard.jsx + FeatureCard.css)
**Purpose:** Individual feature display card
**Props:**
- title, description, icon, type, hasImage, badge, icons, gradient, imageHeight
**Variants:**
- Large cards with images
- Cards with icon boxes
- Cards with badges
**Features:**
- Hover effects with border glow
- Flexible content based on props

### 6. Experience (Experience.jsx + Experience.css)
**Purpose:** Showcase platform capabilities
**Contains:**
- Two-column layout
- Left: Text content explaining the platform
- Right: Visual cards showing "Pattern Generation" and "Emotional Analysis"
- Hover effects on analysis cards

### 7. Founder (Founder.jsx + Founder.css)
**Purpose:** Team/founder introduction
**Contains:**
- Two-column layout
- Left: Founder card with avatar, name, role, badge
- Right: Content explaining "by producers, for producers"
- Carnegie Mellon mention

### 8. Trusted (Trusted.jsx + Trusted.css)
**Purpose:** Display partner/tool logos
**Contains:**
- Title "Trusted by Leading Music Producers"
- Horizontal list of brand names (Logic, Ableton, FL Studio, etc.)
- Hover effects on logos

### 9. Testimonial (Testimonial.jsx + Testimonial.css)
**Purpose:** Customer/user testimonial
**Contains:**
- Large quote text
- Author avatar and info (Jane Doyle, Producer)
- Centered card layout

### 10. Footer (Footer.jsx + Footer.css)
**Purpose:** Site footer
**Contains:**
- Copyright notice
- Navigation links (Docs, Blog, Support)
- Responsive layout (stacks on mobile)

---

## 🎨 Styling Strategy

### CSS Variables (in GlobalStyles.css)
```
--primary-orange: #ff6b00
--primary-blue: #2563eb
--primary-purple: #6366f1
--bg-dark, --bg-mid, --bg-light
--text-primary, --text-secondary, etc.
```

### Component CSS Pattern
Each component has its own CSS file with:
- Component-specific classes (prefixed with component name)
- Hover/active states
- Responsive breakpoints
- Animations

### Animations
Three main animation types defined in GlobalStyles.css:
- `fadeInDown` - Header animation
- `fadeInUp` - Section animations
- `fadeInScale` - Arpeggiator demo animation

Each section has staggered animation delays for smooth sequential reveals.

---

## 🔄 Data Flow

```
App.jsx (Root)
  ├── Header.jsx
  ├── Hero.jsx
  │     └── ArpeggiatorDemo.jsx (manages note state)
  ├── Features.jsx
  │     └── FeatureCard.jsx (x4, receives props)
  ├── Experience.jsx
  ├── Founder.jsx
  ├── Trusted.jsx
  ├── Testimonial.jsx
  └── Footer.jsx
```

- **Props flow down** from Features to FeatureCard
- **State managed locally** in ArpeggiatorDemo for note selection
- **No global state** - each component is self-contained

---

## 📱 Responsive Design

All components include responsive breakpoints:
- Desktop: Full width up to 1400px container
- Tablet (≤1024px): Grid layouts become single column
- Mobile (≤768px): 
  - Navigation hidden
  - Note grid becomes 6 columns
  - All grids become single column
  - Reduced padding

---

## 🚀 Quick Start Commands

```bash
npm install          # Install dependencies
npm run dev          # Start development server
npm run build        # Production build
npm run preview      # Preview production build
```

---

## 🔧 Customization Guide

### Change Colors
Edit CSS variables in `src/styles/GlobalStyles.css`

### Add New Section
1. Create `NewSection.jsx` and `NewSection.css` in `src/components/`
2. Import and add to `App.jsx`
3. Use existing components as templates

### Modify Features
Edit the `features` array in `Features.jsx`

### Update Content
Each component has hardcoded content - edit directly in JSX

---

## 📦 Dependencies

- **react** & **react-dom** - UI framework
- **lucide-react** - Icon library
- **vite** - Build tool & dev server
- **@vitejs/plugin-react** - React support for Vite

No heavy dependencies - lightweight and fast!
