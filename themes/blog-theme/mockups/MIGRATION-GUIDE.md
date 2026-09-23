# blog.can.ac Style → saidrassai.github.io Migration Guide

## Key Style Elements Identified on blog.can.ac

### 1. Color Scheme
- **Dark theme**: Near-black background (`#070708` or similar)
- **Text**: Light gray on dark (`#e4e6e8`)
- **Accent**: Red/pink numbering badges (`#e11d47` — same red your current theme uses for the favicon)
- **Category tags**: Color-coded pills with distinct hues per category (blue, orange, purple, green tints)
- **Borders**: Very subtle (`rgba(255,255,255,0.03-0.04)`)

### 2. Typography
- **Monospace primary font**: Fira Code / JetBrains Mono / SF Mono (not Georgia/serif)
- **Very dense**: `font-size: 0.82-0.87rem`, `line-height: 1.45`
- **No large hero headers** on the listing page
- **Compact headings**: `h1` at ~1.5rem, `h2` at ~0.95rem, not the large 2.4rem your current theme uses

### 3. Layout Philosophy
- **Maximum information density** — everything is squeezed into narrow vertical space
- **No wasted whitespace** — margins between articles are 0.9rem (your current: 2rem)
- **Two-column grid on listing**: `[N]` number badge | title + metadata
- **No hero images on listing** — articles are represented by text only
- **No "card" hover effects** — subtle `rgba(255,255,255,0.015)` background change on hover

### 4. Article Listing Structure (homepage)
```
[13]  The Minutiae of Tool-calling          ← sequential number + title on one row
      CONTEXT CARRIER / 1568² PNG ...      ← category tag + dense tech metadata
      Software engineering 10 min           ← date + read time
```

Key differences from your current layout:
- Your current: 4-column grid (Number | Article | Summary | Author)
- blog.can.ac: 2-column (Number | Title + metadata) — much more compact
- blog.can.ac has **inline technical metadata** (like `1568² PNG`, `P0`, `10 min`) packed into the listing row
- Sequential numbering in colored badges (not just "001", "002")

### 5. Article Pages
- **Category link** above title (e.g., `Software engineering — 03.08.2026`)
- **Date format**: `DD.MM.YYYY` (not "Jan 2, 2006")
- **og:image** (link preview image) shown at top
- **Section numbering**: `## 0x0: Title`, `## 0x1: Title` (like x86 register offsets)
- **Dense body text**: 0.82rem, line-height 1.55
- **No TOC sidebar** — content is full-width
- **Bottom navigation**: `[01 title→` / `← title [13]` style

### 6. Footer / Author Bio
- Small avatar image (48×48px circle)
- Author name + description below
- Nav links to prev/next article in `[NN` / `NN]` format

### 7. Header / Nav
- **Logo only** (no "Blog" link, no GitHub icon in nav)
- **Social links** as compact buttons (GitHub, X, LinkedIn, HackerOne)
- **Search** field labeled "Search the index"
- **Tagline** below: description + author name + "N articles — YYYY–2026"

## Concrete Changes Needed for Your Hugo Theme

### File: `themes/blog-theme/layouts/partials/header.html`
- Change `"rassai."` → `"can.ac"` styling (just name, no nav links)
- Add social link buttons (GitHub, X, LinkedIn) in compact style
- Add search field (or "Search the index" text)

### File: `themes/blog-theme/assets/main.css`
- **Switch background** from `#faf8f7` (light) to `#070708` (dark)
- **Switch text** from `#021117` to `#e4e6e8`
- **Switch font-family** from `Georgia,serif` to `monospace` stack
- **Adjust font sizes**: h1 1.5rem, h2 0.95rem, body 0.82rem
- **Adjust line-height**: 1.45-1.55
- **Add category tag pill styles** with color variants
- **Add dark code block** styles (already partially present)

### File: `themes/blog-theme/layouts/partials/posts_list.html`
- **Replace 4-column grid** with 2-column (number | title+meta)
- **Reduce padding** from 2rem to 0.9rem between articles
- **Add inline technical metadata** support via frontmatter params
- **Add category tags** from `.Params.categories` or `.Params.tags`
- **Change date format** to `02.01.2006`
- **Change numbering** to colored badges like `[01]`, `[02]`, `[03]`

### File: `themes/blog-theme/layouts/_default/single.html`
- **Add category link** above title
- **Change date format** to `DD.MM.YYYY`
- **Add section numbering** (0x0, 0x1, etc.) — requires markdown heading customization
- **Remove TOC sidebar** (or make it optional)
- **Add prev/next nav** in `[NN title→` format
- **Add author bio footer** with avatar

### File: `hugo.toml`
- Add `markup.highlight` style for dark theme (already set to 'github' — switch to 'nord' or custom)

## Mockup Files
- `mockups/blog-canac-style-mockup.html` — Standalone HTML mockup (viewable in browser)
  - Contains: homepage listing + commented article page template
  - Open in browser to see the visual style

## Your Articles Adapted to the Style

| Number | Title | Category | Date | Tech Metadata |
|--------|-------|----------|------|---------------|
| [03] | Building an ENTERPRISE CPU RAG RESEARCH 2025–2026 | RAG | 05.06.2026 | CPU-ONLY / BGE-M3 + SPLADEv3 + COLBERT-v2 / FINANCEBENCH 8K |
| [02] | Building a Finance Agent and Dataset: From Research Note to Replication | LLM | 02.06.2026 | Fin-R1-DATA / DEEPSEEK-R1-671B / QWEN2.5-72B |
| [01] | Fine-tuning LFM2.5-1.2B-Instruct with GRPO | SYSTEMS | 15.06.2026 | LFM2.5-1.2B / GRPO / UNSLOTH / OCR INVOICE EXTRACTION |
