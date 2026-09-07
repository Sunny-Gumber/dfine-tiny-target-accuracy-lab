import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import FaceLab from "./FaceLab";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");

const params = new URLSearchParams(window.location.search);
const page = params.get("lab") === "face" ? <FaceLab /> : <App />;

createRoot(root).render(
  <StrictMode>
    {page}
  </StrictMode>,
);
