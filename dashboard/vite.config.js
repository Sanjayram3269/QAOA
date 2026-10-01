import { defineConfig } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dashboardDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(dashboardDir, '..');

export default defineConfig({
  root: repoRoot,
  server: {
    port: 5173,
    strictPort: false,
    fs: { allow: [repoRoot] }
  },
  build: {
    outDir: path.resolve(dashboardDir, 'dist'),
    emptyOutDir: true,
    rollupOptions: { input: path.resolve(dashboardDir, 'index.html') }
  }
});
