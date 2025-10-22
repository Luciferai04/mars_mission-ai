# TikZ Architecture Diagrams Guide

This document explains how to compile and use the TikZ architecture diagrams for the Mars Mission AI system.

## Overview

All architecture diagrams are available in TikZ/LaTeX format in `docs/tikz_diagrams.tex`. TikZ provides publication-quality vector graphics suitable for academic papers, technical documentation, and presentations.

## Available Diagrams

The file contains 18 comprehensive diagrams:

1. **High-Level System Architecture**
   - Shows all microservices, gateway, data layer, and ML models
   - Client-to-service flow
   - Service dependencies

2. **MARL System Architecture**
   - 5 specialized RL agents
   - Weighted voting coordination
   - Learning loop with Q-tables and experience replay

3. **Data Flow Sequence Diagram**
   - Request flow from client through all services
   - Synchronous service calls
   - Response propagation

4. **Docker Deployment Architecture**
   - Complete Docker container layout
   - Network topology
   - Volume mappings
   - Service layers (Gateway, Service, Data, Volumes)

5. **MARL Agent State-Action Space**
   - State machine diagram
   - RL training loop
   - State transitions and rewards

6. **MARL Training Performance**
   - Training curve graph
   - Convergence visualization
   - Performance metrics over episodes

7. **Service Response Times**
   - Bar chart of P95 latencies
   - Comparative performance across services

8. **Microservices Communication Pattern**
   - Service-to-service communication
   - Service discovery
   - Message queue integration

9. **DQN-Enabled MARL (Double DQN)**
   - Policy/Target networks, replay buffer, weighted voting

10. **Multi-Rover Coordination**
    - Fleet coordinator with conflict resolution

11. **Predictive Maintenance Flow**
    - Telemetry → features → model → risk score

12. **Streaming Telemetry (WebSocket)**
    - Publisher, WS hub, subscribers (mobile/web)

13. **Voice Command Flow**
    - Voice service to planning

14. **Federated Learning (FedAvg)**
    - Clients send weights, aggregator returns global model

15. **3D Terrain Visualization**
    - DEM → /heightmap → Three.js viewer

16. **Long-Term Strategic Planning**
    - Mission context → weekly objectives over horizon

17. **JPL Export (APGEN/PLEXIL)**
    - Plan JSON → exporters

18. **Mobile App Architecture (Expo)**
    - Expo app consuming WS and REST APIs

## Prerequisites

### macOS

```bash
# Install MacTeX (full TeX distribution)
brew install --cask mactex

# Or install BasicTeX (minimal)
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install standalone tikz pgf
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install texlive-latex-base texlive-pictures
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install texlive-latex texlive-pgf
```

### Windows

Download and install [MiKTeX](https://miktex.org/download) or [TeX Live](https://www.tug.org/texlive/)

## Compilation

### Compile All Diagrams (Multi-page PDF)

```bash
cd docs
pdflatex tikz_diagrams.tex
```

This creates `tikz_diagrams.pdf` with all 8 diagrams on separate pages.

### Compile Individual Diagrams

To extract specific diagrams, create individual files:

**Example: High-Level Architecture Only**

```latex
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds, shadows, calc}

\begin{document}

% Copy the "High-Level System Architecture" section from tikz_diagrams.tex
% Paste here

\end{document}
```

Save as `architecture_highlevel.tex` and compile:

```bash
pdflatex architecture_highlevel.tex
```

### Convert to PNG (High Resolution)

```bash
# Install ImageMagick
brew install imagemagick  # macOS
sudo apt-get install imagemagick  # Linux

# Convert each page to PNG
pdftoppm tikz_diagrams.pdf diagram -png -r 300

# This creates: diagram-1.png, diagram-2.png, etc.
```

### Convert to SVG (Vector Format)

```bash
# Install pdf2svg
brew install pdf2svg  # macOS
sudo apt-get install pdf2svg  # Linux

# Convert specific page
pdf2svg tikz_diagrams.pdf diagram1.svg 1
```

## Usage in Documentation

### LaTeX Papers/Thesis

```latex
\documentclass{article}
\usepackage{graphicx}

\begin{document}

\section{System Architecture}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{tikz_diagrams.pdf}
    \caption{Mars Mission AI Architecture}
    \label{fig:architecture}
\end{figure}

\end{document}
```

### Markdown/GitHub (via PNG conversion)

```markdown
## Architecture

![System Architecture](diagrams/architecture.png)
```

### PowerPoint/Keynote

1. Compile to PDF
2. Open PDF in Preview/Acrobat
3. Export as PNG at 300 DPI
4. Insert into presentation

### Web Documentation

```html
<img src="diagrams/architecture.svg" alt="System Architecture" />
```

## Customization

### Change Colors

Edit the color definitions at the top of `tikz_diagrams.tex`:

```latex
\definecolor{planning}{RGB}{74,144,226}  % Blue
\definecolor{vision}{RGB}{80,200,120}    % Green
\definecolor{marl}{RGB}{255,107,107}     % Red
\definecolor{data}{RGB}{255,217,61}      % Yellow
\definecolor{gateway}{RGB}{0,117,143}    % Teal
```

### Modify Layout

Adjust node positioning:

```latex
% Change spacing
\node (service) [below=3cm of gateway] {Service};

% Change absolute position
\node (service) at (5,4) {Service};
```

### Add/Remove Elements

```latex
% Add new service
\node (newservice) [service, fill=blue!30] at (x,y) {New Service};

% Add connection
\draw [arrow] (source) -- (destination);
```

## Export Formats

| Format | Use Case | Command |
|--------|----------|---------|
| PDF | Documents, papers | `pdflatex tikz_diagrams.tex` |
| PNG | Web, presentations | `pdftoppm -png -r 300` |
| SVG | Web, scaling | `pdf2svg` |
| EPS | LaTeX inclusion | `pdftops -eps` |
| JPEG | Compressed images | `convert -quality 95` |

## Advanced Features

### Include in Overleaf

1. Upload `tikz_diagrams.tex` to Overleaf project
2. Include in main document:

```latex
\input{tikz_diagrams}
```

### Batch Processing

```bash
#!/bin/bash
# compile_diagrams.sh

for i in {1..8}; do
    pdf2svg tikz_diagrams.pdf diagram_$i.svg $i
    pdftoppm tikz_diagrams.pdf diagram_$i -png -r 300 -f $i -l $i
done
```

### Animation (Beamer)

```latex
\documentclass{beamer}
\usepackage{tikz}

\begin{document}

\begin{frame}{System Evolution}
    \only<1>{% Initial architecture}
    \only<2>{% With MARL}
    \only<3>{% Complete system}
\end{frame}

\end{document}
```

## Troubleshooting

### Missing TikZ Libraries

```bash
sudo tlmgr install pgf tikz-positioning tikz-arrows
```

### Font Issues

```bash
# Use standard fonts
\usepackage{lmodern}

# Or embed all fonts
pdflatex -output-format=pdf tikz_diagrams.tex
```

### Compilation Errors

```bash
# Clean auxiliary files
rm *.aux *.log *.out

# Recompile
pdflatex tikz_diagrams.tex
```

### Large File Size

```bash
# Compress PDF
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile=compressed.pdf tikz_diagrams.pdf
```

## Best Practices

1. **Version Control**: Keep `.tex` source in Git, generate PDFs on-demand
2. **High DPI**: Use 300 DPI or higher for print quality
3. **Vector First**: Prefer PDF/SVG for documentation
4. **Consistent Style**: Maintain color scheme across all diagrams
5. **Annotations**: Add labels and legends for clarity
6. **Accessibility**: Ensure sufficient color contrast

## Integration with CI/CD

### GitHub Actions

```yaml
name: Generate Diagrams

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install LaTeX
        run: sudo apt-get install texlive-pictures
      - name: Compile diagrams
        run: |
          cd docs
          pdflatex tikz_diagrams.tex
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: diagrams
          path: docs/tikz_diagrams.pdf
```

## Resources

- [TikZ Manual](https://tikz.dev/)
- [TikZ Examples](https://texample.net/tikz/)
- [PGF/TikZ Gallery](https://www.overleaf.com/gallery/tagged/pgf-tikz)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX/PGF/TikZ)

## Quick Reference

```bash
# Single command to generate all formats
pdflatex tikz_diagrams.tex && \
for i in {1..18}; do \
    pdftoppm tikz_diagrams.pdf diagram_$i -png -r 300 -f $i -l $i; \
    pdf2svg tikz_diagrams.pdf diagram_$i.svg $i; \
done
```

This generates:
- `tikz_diagrams.pdf` (all diagrams)
- `diagram_1-1.png` through `diagram_18-1.png` (PNG images)
- `diagram_1.svg` through `diagram_18.svg` (SVG vectors)

---

**Note**: All diagrams are publication-ready and suitable for academic papers, technical reports, and presentations.
