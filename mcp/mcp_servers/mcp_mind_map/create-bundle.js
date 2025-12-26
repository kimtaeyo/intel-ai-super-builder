#!/usr/bin/env node
/**
 * Bundle markmap-cli for pkg
 * Converts ESM to CommonJS and handles import.meta.url
 */

const esbuild = require('esbuild');
const path = require('path');
const fs = require('fs');

const bundleDir = path.join(__dirname, 'bundled');

// Clean bundled directory
if (fs.existsSync(bundleDir)) {
  fs.rmSync(bundleDir, { recursive: true, force: true });
}
fs.mkdirSync(bundleDir, { recursive: true });

console.log('Bundling markmap-cli with esbuild...');

// Determine entry point - new versions use cli.js, old versions use index.js
let entryPoint;
try {
  entryPoint = require.resolve('markmap-cli/dist/cli.js');
  console.log('Using CLI entry point: dist/cli.js');
} catch {
  entryPoint = require.resolve('markmap-cli/dist/index.js');
  console.log('Using legacy entry point: dist/index.js');
}

esbuild.build({
  entryPoints: [entryPoint],
  bundle: true,
  platform: 'node',
  target: 'node18',
  format: 'cjs',
  outfile: path.join(bundleDir, 'index.js'),
  // Bundle everything - no external packages
  banner: {
    js: `
// Shim for import.meta.url in CommonJS
const __import_meta_url__ = require('url').pathToFileURL(__filename).href;
const __import_meta_dirname__ = __dirname;
`
  },
  define: {
    'import.meta.url': '__import_meta_url__',
    'import.meta.dirname': '__import_meta_dirname__'
  },
  mainFields: ['module', 'main'],
  logLevel: 'info',
}).then(() => {
  console.log('✓ Bundle created successfully in bundled/index.js');
  const stats = fs.statSync(path.join(bundleDir, 'index.js'));
  console.log(`  Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
}).catch((error) => {
  console.error('✗ Bundle failed:', error);
  process.exit(1);
});
