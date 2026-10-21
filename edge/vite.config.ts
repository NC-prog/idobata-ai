import { defineConfig } from "vite";

// ビルド成果物はフロントAI（FastAPI、server/src/edge_server.py）が /edge/ で配信する
export default defineConfig({
  base: "/edge/",
  build: {
    outDir: "../server/static/edge",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
});
