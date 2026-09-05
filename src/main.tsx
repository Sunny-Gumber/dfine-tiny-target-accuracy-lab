import { createRoot } from "react-dom/client";
import Home from "../app/page";
import "../app/globals.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Application root was not found.");
}

createRoot(root).render(
  <>
    <Home />
    <a
      href="./live.html"
      style={{
        position: "fixed",
        right: 18,
        bottom: 18,
        zIndex: 80,
        border: "1px solid rgba(103,232,249,.35)",
        borderRadius: 999,
        background: "rgba(8,47,73,.94)",
        color: "#cffafe",
        padding: "10px 15px",
        fontSize: 12,
        fontWeight: 800,
        boxShadow: "0 12px 40px rgba(0,0,0,.35)",
        textDecoration: "none",
      }}
    >
      ⚡ 50 ms Live
    </a>
  </>,
);
