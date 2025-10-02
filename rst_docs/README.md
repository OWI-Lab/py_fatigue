# Sphinx Documentation

This directory contains the Sphinx documentation source files for py-fatigue.

## Building Documentation Locally

To build the documentation locally, use the invoke tasks:

```bash
# Clean and rebuild documentation
inv docs.rebuild

# Serve documentation with hot reload
inv docs.autobuild

# Clean build directory
inv docs.clean
```

## CI/CD Publishing

The documentation is automatically built and published using GitHub Actions:

- **Workflow**: `.github/workflows/pages.yml`
- **Trigger**: On push to any branch (for testing), but only deploys from `develop`
- **Deployment**: Publishes to `gh-pages` branch
- **URL**: https://owi-lab.github.io/py_fatigue/

### How it Works

1. On every push/PR, the workflow builds Sphinx documentation to verify it compiles correctly
2. Only when pushing to the `develop` branch, the workflow deploys the built HTML to `gh-pages`
3. GitHub Pages serves the documentation from the `gh-pages` branch

### GitHub Pages Configuration

To ensure GitHub Pages is properly configured:

1. Go to repository Settings → Pages
2. Ensure "Source" is set to "Deploy from a branch"
3. Select branch: `gh-pages` and folder: `/ (root)`
4. The documentation will be available at https://owi-lab.github.io/py_fatigue/

The `sphinx.ext.githubpages` extension (enabled in `conf.py`) automatically creates a `.nojekyll` file to ensure GitHub Pages serves Sphinx documentation correctly.