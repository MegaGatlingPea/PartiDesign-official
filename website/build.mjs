import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const output = join(root, "dist");
const assetRoot = join(root, "..", "docs", "assets");
const assets = [
  "milkfrog.png", "model-overview.png", "capabilities/protein.png",
  "capabilities/nucleic-acid.png", "capabilities/ligand.png", "capabilities/pipeline.png",
];

// Check curated assets before replacing a previous build.
await Promise.all(assets.map((asset) => readFile(join(assetRoot, asset))));
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
const css = spawnSync(process.execPath, [
  join(root, "node_modules", "@tailwindcss", "cli", "dist", "index.mjs"),
  "--input", join(root, "styles.css"), "--output", join(output, "styles.css"), "--minify",
], { cwd: root, stdio: "inherit" });
if (css.error) throw css.error;
if (css.status !== 0) process.exit(css.status ?? 1);
await Promise.all([
  ...["index.html", "script.js", "robots.txt", "sitemap.xml"].map((file) =>
    copyFile(join(root, file), join(output, file))),
  ...assets.map(async (asset) => {
    const destination = join(output, "assets", asset);
    await mkdir(dirname(destination), { recursive: true });
    await copyFile(join(assetRoot, asset), destination);
  }),
  writeFile(join(output, ".nojekyll"), ""),
]);
console.log(`Built static site in ${output} with ${assets.length} selected assets.`);
