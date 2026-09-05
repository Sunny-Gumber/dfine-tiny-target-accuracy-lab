import { createRoot } from "react-dom/client";
import SmartPage from "../app/smart-page";
import "../app/globals.css";

const root = document.getElementById("root");
if (!root) throw new Error("Application root was not found.");
createRoot(root).render(<SmartPage />);
