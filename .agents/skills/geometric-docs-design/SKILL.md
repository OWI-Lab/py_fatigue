---
name: geometric-docs-design
description: Design and review documentation websites using a geometric visual system derived from zed.dev. Use for documentation homepages, guides, API references, and their shared navigation, tabs, examples, and footers.
---

# Geometric Documentation Design

Use the geometry to show readers where they are, how topics relate, and where an example begins. The same visual system should work for a short quickstart, a long guide, and a dense API reference.

## Start with the existing site

Inspect the documentation home, a representative guide, and a dense reference page. Identify the shared shell, theme tokens, navigation, and content components. Reuse and extend those components. Put shared styles in the documentation theme and page-specific layout in the relevant template.

In the repository, read `DESIGN.md` for the palette and typography. Use `data_api/static/css/docs-overview.css` for the landing frame, `data_api/static/css/geometric-design.css` for controls, and `data_api/static/frontend/dashboards/css/ruled-workspace.css` for ruled sidebars and tabs. On another stack, map those roles to the site’s existing tokens and components.

Use Delta’s documentation as the reference for sidebar indexes and theme placement. Use Zed and Delta as references for alignment, framing, and restrained pattern. Use design own palette, typography, marks, and exact SCADA tabs. [Delta Docs](https://delta.dev/docs/getting-started) · [Zed](https://zed.dev/) · [Zed Brand](https://zed.dev/brand)

## Build one continuous frame

- Align the masthead, content sections, and footer to shared vertical rails. Carry the rails through the footer art to the end of the page, including on short pages. Give prose a comfortable reading measure and let wide examples and reference tables use the available article width.
- Derive quiet 1px rules from `--color-border`. Extend major rules past the centered content into the available gutters. Draw each seam once and carry it continuously across the inner columns.
- Place a small, outlined 6px diamond with barely rounded corners where section rules meet the rails. Keep the section edges square and flush. Use small outlined circles at the footer and artwork seams.
- Use hairlines within a topic and a shallow, tinted band with faint diagonal hatching at major transitions. Give selected hairlines solid ends and a short, densely dashed center. A compact divider about one-third of the standard band height works beneath an embedded demo.
- Alternate `--color-base-100` and `--color-base-200` by section purpose. Tint introductions, section overviews, and selected groups of examples. Give reading passages, code, tables, and controls plain surfaces.
- Use a faint square grid behind selected headings and sidebar headers. In a metadata strip, let it fade into the last visible cell and reach the outer edge. At selected bands, place graduated horizontal ruler ticks just outside the rails. Vary their lengths and opacity along the edge.
- Use square grids, graduated ruler ticks, and shallow diagonal divider bands across the documentation. Reserve wave and wind textures for landing-page gutters and decorative artwork for the footer. Render patterns as non-interactive CSS layers and keep their contrast low in both themes.

## Sidebar indexes and theme placement

- Use three columns on wide screens, following Delta Docs. Put the documentation index on the left, the article in the center, and **On this page** on the right. Align them to the shared frame. Keep both indexes in view as the reader scrolls.
- Organize the left index into named topic groups with plain page links. Use full-width disclosure rows and hairline separators. Give the index heading a tinted surface, a fading grid, and small diamond junctions at its seams. Open the group containing the current page. Preserve the reader’s disclosure choices within the section and scroll the index independently when needed.
- Build the right index from article headings with stable anchors. Indent nested headings and highlight the section currently in view. Keep this index visually quieter than the left one. Give heading targets enough scroll offset to clear the sticky masthead.
- Put the compact **Light / Dark / System** selector next to Search at the right end of the desktop masthead. Repeat it as a full-width, three-segment utility row at the foot of the left sidebar. Place the row outside the scrolling index with one hairline above it. Both placements share one persisted preference and expose the same selected state and accessible labels.
- As space narrows, move **On this page** into a disclosure near the article title. Move the left index into a navigation drawer and keep the theme selector in the drawer’s bottom utility row. Escape closes the drawer and returns focus to its trigger.

## Use the exact SCADA tabs

- Reuse the ruled tabs from SCADA Reporting at `app.24sea.localhost/reporting/scada/` for documentation tabs, including language and platform examples. Use `view-tabs.html` with `ruled=True`, the shared `tabs.css`, the `.dashboard-tabbar--ruled` rules in `ruled-workspace.css`, and the shared `tabs.js`. On another stack, port the rendered markup and CSS together. Treat SCADA as the visual reference for the port.
- Preserve the 28px strip, square edges, regular-weight labels, and shared canvas background. The active tab has a 1px top rule and opens into its panel at the baseline. Inactive labels are muted. The first tab has a clean leading edge. Each subsequent tab starts with a short vertical rule and a hollow 4px circle, alternating top and bottom.
- Preserve the app’s hover and focus states, label outline, responsive padding, and horizontal scrolling that keeps the active tab in view. Use Arrow Left and Arrow Right to move between tabs and Home and End to reach the first and last. Keep tab state shareable and compatible with browser history.

## Adapt the frame to the page

- **Documentation home.** Use a short gridded introduction and flush destination cells for the main reader tasks. Give each cell a clear destination and enough context to choose it. Use divider bands to mark transitions between genuinely different sections.
- **Guide.** Place the title and any prerequisites above the reading column. Align figures, examples, and admonitions to the article grid. Keep steps and examples in reading order as the layout stacks.
- **API reference.** Use compact ruled strips for real metadata such as the HTTP method, endpoint, authentication, and version. Keep the method and path easy to scan. Use plain tables and code surfaces for parameters, responses, and examples. Place examples beside their explanations when both fit comfortably and immediately after them when they stack.

## Typography, examples, and controls

- Use IBM Plex Serif at weight 400 for headings, IBM Plex Sans for prose, IBM Plex Sans Condensed for navigation and compact labels, and IBM Plex Mono for code. Give prose a comfortable line height. Use the condensed face where density helps scanning.
- Use Euler for inline and display mathematics. With MathJax 4, load the Euler extension and pair it with a compatible base font such as Pagella for glyph coverage. Verify the Euler font loads and equations render in the browser.
- Frame code examples with a single hairline and square edges. Use a small toolbar for a filename or language label when useful, with the copy action at the top right. Use theme-aware syntax colors, with Stack Overflow–inspired palette as the reference. Copy the runnable text without decorative prompts or line numbers. Contain horizontal scrolling inside long code blocks and wide tables.
- Give admonitions square corners, a soft semantic fill, a 1px outline, and a thicker left rule. Identify warnings through wording as well as color. Style disclosures as quiet ruled rows.
- Reuse shared controls for search, selectors, and forms. Give fields and switch tracks 4px corners, use a squared switch thumb, and join adjacent controls with a shared hairline. Use the platform action fill for selected controls and `--frontend-control-shadow` for framed actions. Keep ordinary links and navigation rows flat. Give sidebar errors compact text, consistent insets, and a theme-aware error color.

## Finish at the footer

Use the footer’s top rule as the final section boundary. Continue the vertical rails through the footer links and artwork. Place site links and legal information in the footer utility row.

For SAIL footer art, bottom-align the official outlined logomark and wordmark over faint, fading rules. Use a consistent, low-contrast stroke treatment for the artwork and surrounding geometry. Mark the rail intersections with small outlined circles.

## Verify the result

Check the documentation home, a long guide, and a dense reference in light and dark mode at desktop and phone widths. Confirm that gutter rules meet the rails cleanly and wide blocks scroll locally. Follow a deep link, navigate the indexes and tabs by keyboard, switch themes from each placement, and copy a runnable example.

Compare the documentation tabs side by side with SCADA at matching viewport widths and in both themes. Check the active outline, baseline, alternating circles, spacing, and focus treatment. When updating a shared component, check both a migrated page and an existing consumer.
