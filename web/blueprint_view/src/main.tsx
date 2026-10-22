import { DimThemeProvider } from "dimos-react";
import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";
import "./view.css";

// Desktop's skins' tokens are at /theme.css (the view is /dimos-gateway/blueprint_view); off Desktop the bundled copy stands in.
createRoot(document.getElementById("root")!).render(
  <DimThemeProvider themeUrl="../theme.css">
    <App />
  </DimThemeProvider>,
);
