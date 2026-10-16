import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/archivo/wdth.css";
import "@fontsource-variable/literata";
import "@fontsource/overpass-mono/400.css";
import "@fontsource/overpass-mono/600.css";
import "./styles/base.css";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
