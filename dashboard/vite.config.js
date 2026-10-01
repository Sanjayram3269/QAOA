import { defineConfig } from 'vite';
import path from 'node:path';

const dashboardDir = __dirname;
const repoRoot = path.resolve(dashboardDir, '..');

export default defineConfig({
  root: repoRoot,
  server: {
    port: 5173,
    strictPort: false,
    fs: {
      allow: [repoRoot]
    }
  },
  build: {
    outDir: path.resolve(dashboardDir, 'dist'),
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(dashboardDir, 'index.html')
    }
  }
});
