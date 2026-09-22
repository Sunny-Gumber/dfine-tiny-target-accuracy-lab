const MODEL_CACHE = "cctv-ai-model-assets-v1";
const CACHE_PREFIX = "cctv-ai-model-assets-";

function isCacheableModelRequest(request) {
  if (request.method !== "GET") return false;

  const url = new URL(request.url);
  const path = url.pathname.toLowerCase();

  // Both YOLOX-S and RelateAnything are fetched as ONNX assets. The
  // RelateAnything predicate bank is NPZ. Keep WASM runtime files too so
  // CPU fallback does not repeatedly fetch them.
  const isModelAsset =
    path.endsWith(".onnx") ||
    path.endsWith(".npz") ||
    path.endsWith(".wasm");

  if (!isModelAsset) return false;

  // Partial responses cannot be safely stored in CacheStorage.
  if (request.headers.has("range")) return false;

  return true;
}

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    Promise.all([
      caches.keys().then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith(CACHE_PREFIX) && key !== MODEL_CACHE)
            .map((key) => caches.delete(key)),
        ),
      ),
      self.clients.claim(),
    ]),
  );
});

async function cacheFirst(request) {
  const cache = await caches.open(MODEL_CACHE);
  const cached = await cache.match(request, { ignoreVary: true });

  if (cached) {
    return cached;
  }

  const response = await fetch(request);

  // Cache only complete successful responses. Opaque responses can also be
  // stored, but the current model hosts expose CORS so we normally get 200.
  if ((response.ok || response.type === "opaque") && response.status !== 206) {
    cache.put(request, response.clone()).catch(() => undefined);
  }

  return response;
}

self.addEventListener("fetch", (event) => {
  if (!isCacheableModelRequest(event.request)) return;
  event.respondWith(cacheFirst(event.request));
});
