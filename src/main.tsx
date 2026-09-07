import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import FaceLabBundled from "./FaceLabBundled";
import FaceLabScrfd from "./FaceLabScrfd";
import "./styles.css";
import "./face.css";

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");

const params = new URLSearchParams(window.location.search);
const isFaceLab = params.get("lab") === "face";
const facePage = params.get("engine") === "blaze" ? <FaceLabBundled /> : <FaceLabScrfd />;
const page = isFaceLab ? facePage : <App />;

createRoot(root).render(
  <StrictMode>
    {page}
  </StrictMode>,
);
