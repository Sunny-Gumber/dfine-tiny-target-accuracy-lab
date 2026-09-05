import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  base: "/dfine-tiny-target-accuracy-lab/",
  plugins: [react()],
  build: {
    outDir: "dist-github",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        live: fileURLToPath(new URL("./live.html", import.meta.url)),
        smart: fileURLToPath(new URL("./smart.html", import.meta.url)),
        shootout: fileURLToPath(new URL("./shootout.html", import.meta.url)),
      },
    },
  },
});
