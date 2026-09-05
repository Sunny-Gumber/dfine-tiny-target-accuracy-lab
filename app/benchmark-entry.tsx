import React from "react";
import { createRoot } from "react-dom/client";
import "./globals.css";
import BenchmarkPage from "./benchmark-page";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BenchmarkPage />
  </React.StrictMode>,
);
