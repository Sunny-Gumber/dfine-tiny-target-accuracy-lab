import React from "react";
import { createRoot } from "react-dom/client";
import "./globals.css";
import VideoPage from "./video-page";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <VideoPage />
  </React.StrictMode>,
);
