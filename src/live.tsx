import { createRoot } from "react-dom/client";
import LivePage from "../app/live-page";
import "../app/globals.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Application root was not found.");
}

createRoot(root).render(<LivePage />);
