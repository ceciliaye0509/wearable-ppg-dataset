/* Only static application files are cached. Task files never pass through fetch. */
const CACHE='daily-hrv-local-review-v1-20261113-grouped-zip-v13';
const ASSETS=['./','./index.html','./styles.css','./core.js','./storage.js','./app.js','./archive.js','./history.js'];
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)).then(()=>self.skipWaiting())));
self.addEventListener('activate',e=>e.waitUntil((async()=>{for(const key of await caches.keys())if(key.startsWith('daily-hrv-local-review-')&&key!==CACHE)await caches.delete(key);await self.clients.claim();for(const c of await self.clients.matchAll())c.postMessage({type:'offline-ready'});})()));
self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;const u=new URL(e.request.url);if(u.origin!==location.origin)return;const paths=new Set(ASSETS.map(p=>new URL(p,location.href).pathname));if(!paths.has(u.pathname)||u.search)return;e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request)));});
