export type ModelCacheStatus = {
  supported: boolean;
  ready: boolean;
  persistent: boolean;
};

const CONTROLLER_WAIT_MS = 2500;

function waitForController() {
  if (navigator.serviceWorker.controller) return Promise.resolve();

  return new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      navigator.serviceWorker.removeEventListener("controllerchange", finish);
      resolve();
    };

    navigator.serviceWorker.addEventListener("controllerchange", finish, { once: true });
    window.setTimeout(finish, CONTROLLER_WAIT_MS);
  });
}

export async function prepareModelCache(): Promise<ModelCacheStatus> {
  if (!("serviceWorker" in navigator) || !("caches" in window)) {
    return { supported: false, ready: false, persistent: false };
  }

  let persistent = false;

  try {
    if (navigator.storage?.persist) {
      persistent = await navigator.storage.persist();
    }
  } catch {
    // Persistence is optional. CacheStorage still works if the browser
    // declines the request; it may simply be more likely to be evicted.
  }

  try {
    const appBase = new URL("./", document.baseURI);
    const serviceWorkerUrl = new URL("model-cache-sw.js", appBase);

    await navigator.serviceWorker.register(serviceWorkerUrl.pathname, {
      scope: appBase.pathname,
    });
    await navigator.serviceWorker.ready;
    await waitForController();
    return { supported: true, ready: true, persistent };
  } catch (error) {
    console.warn("Persistent model cache could not be initialized.", error);
    return { supported: true, ready: false, persistent };
  }
}
