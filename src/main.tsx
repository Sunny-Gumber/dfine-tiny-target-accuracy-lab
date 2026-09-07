import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import FaceLabBundled from "./FaceLabBundled";
import FaceLabScrfd from "./FaceLabScrfd";
import FaceLabStableAge from "./FaceLabStableAge";
import "./styles.css";
import "./face.css";

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");

const params = new URLSearchParams(window.location.search);
const isFaceLab = params.get("lab") === "face";
const engine = params.get("engine");
const facePage = engine === "blaze" ? <FaceLabBundled /> : engine === "single-age" ? <FaceLabScrfd /> : <FaceLabStableAge />;
const page = isFaceLab ? facePage : <App />;

createRoot(root).render(
  <StrictMode>
    {page}
  </StrictMode>,
);
