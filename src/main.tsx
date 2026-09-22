import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./runtime";
import App from "./App";
import { prepareModelCache } from "./modelCache";
import "./styles.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found");
}

async function bootstrap() {
  // Register and activate the model-cache service worker before App mounts.
  // This gives the first YOLOX/RelateAnything download a chance to be stored
  // so later visits can reuse the local copy instead of downloading it again.
  await prepareModelCache();

  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

void bootstrap();
