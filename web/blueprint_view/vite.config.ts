/// <reference types="vitest/config" />
import deno from "@deno/vite-plugin";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The gateway serves index.html at /dimos/blueprint_view and the rest under
// /dimos/blueprint_view/, so assets are relative and live in blueprint_view/.
// deno() resolves the jsr packages (dimos-react, graph-view in deno.json); dedupe
// keeps them on the app's one React.
export default defineConfig({
  plugins: [deno(), react()],
  base: "./",
  resolve: { dedupe: ["react", "react-dom"] },
  server: { proxy: { "/dimos": "http://127.0.0.1:5555", "/api": "http://127.0.0.1:5555" } },
  build: { target: "es2022", assetsDir: "blueprint_view" },
  test: {
    environment: "node",
    include: ["src/**/*.test.{ts,tsx}"],
    pool: "threads",
  },
});
