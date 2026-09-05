import React from "react";
import { createRoot } from "react-dom/client";
import "./globals.css";
import ShootoutPage from "./shootout-page";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ShootoutPage />
  </React.StrictMode>,
);
