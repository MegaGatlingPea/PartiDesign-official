# PartiDesign project website

An original static project page built with Tailwind CSS 4 and daisyUI 5. The
production output uses local compiled CSS and JavaScript, with no CDN runtime.

From the repository root, with Node.js 22 or newer:

```bash
cd website
npm ci
npm run build
python3 -m http.server 8000 --directory dist
```

Open `http://localhost:8000`. The build writes `website/dist/` and copies only
the six selected source assets from `docs/assets/`; it does not copy the
repository. `node_modules/` and `dist/` are ignored.

Source files are `index.html`, `styles.css`, and `script.js`. The lightweight
`build.mjs` compiles Tailwind/daisyUI and copies the static files. In-page asset
URLs are relative so the output works under the GitHub project path. Canonical
and social URLs target
`https://megagatlingpea.github.io/PartiDesign-official/`.


The page opens in light mode on a first visit, including when the operating system
prefers dark mode. The header's keyboard-accessible sun/moon button switches the
custom daisyUI themes and remembers an explicit choice in localStorage. An early
head script restores that choice before styles paint; unavailable storage leaves
the toggle usable for the current page. Scientific figure colors remain unchanged.
The theme switch respects reduced-motion settings.

The quickstart recommends the default uv installer and activates `.venv`. The full
uv route still needs Conda or Mamba for native dependencies; the installer prepares
that native prefix automatically. The linked environment guide covers CUDA 13 and
the explicit Conda installer.

After the website and workflow have been committed and pushed to the official
repository, open **Settings → Pages → Build and deployment → Source** and
select **GitHub Actions**. Run the repository's Pages workflow, or let its
configured branch trigger deploy the built `website/dist/` artifact. Building
locally does not publish the site.
