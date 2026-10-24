import { DimThemeProvider } from "dimos-react";
import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";
import "./view.css";

// Desktop's skins' tokens are at /theme.css. The view is served at /gateways/dimos/blueprint_view, with no trailing
// slash, so a relative url resolves against /gateways/dimos/: two levels up, not one. Off Desktop the bundled copy stands in.
createRoot(document.getElementById("root")!).render(
  <DimThemeProvider themeUrl="../../theme.css">
    <App />
  </DimThemeProvider>,
);
