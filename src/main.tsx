import { createRoot } from "react-dom/client";
import FastDemoPage from "../app/fast-demo-page";
import "../app/globals.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Application root was not found.");
}

createRoot(root).render(<FastDemoPage />);
