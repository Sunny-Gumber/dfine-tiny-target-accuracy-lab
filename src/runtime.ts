import { configureRuntime } from "libreyolo-web";

// Configure ONNX Runtime before LibreYOLO creates a session.
// Pin the WASM assets to the installed ORT version and avoid the proxy worker,
// which is less reliable on static GitHub Pages deployments.
configureRuntime({
  wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/",
  numThreads: 1,
  simd: true,
  proxy: false,
});
