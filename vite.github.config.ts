import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/dfine-tiny-target-accuracy-lab/",
  plugins: [react()],
  build: {
    outDir: "dist-github",
    emptyOutDir: true,
  },
});
