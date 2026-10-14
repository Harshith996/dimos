/// <reference types="vitest/config" />
import deno from "@deno/vite-plugin";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The gateway serves index.html at /blueprint_view and the rest under /blueprint_view/ (Desktop puts both
// under /dimos), so assets are relative and live in blueprint_view/. The dev proxy strips /dimos like Desktop.
// deno() resolves the jsr packages (dimos-react in deno.json); dedupe
// keeps them on the app's one React.
export default defineConfig({
  plugins: [deno(), react()],
  base: "./",
  resolve: { dedupe: ["react", "react-dom"] },
  server: {
    proxy: {
      "/dimos": {
        target: "http://127.0.0.1:5555",
        rewrite: (path) => path.replace(/^\/dimos/, ""),
      },
      "/api": "http://127.0.0.1:5555",
    },
  },
  build: { target: "es2022", assetsDir: "blueprint_view" },
  test: {
    environment: "node",
    include: ["src/**/*.test.{ts,tsx}"],
    pool: "threads",
  },
});
