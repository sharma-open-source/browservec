// Ambient declaration for Vite's inlined-worker import (`?worker&inline`), so
// `tsc` accepts the dynamic import in quant/encoder.ts. Vite rewrites the module
// to a base64-inlined Worker constructor at build time; here we just give it a type.
declare module '*?worker&inline' {
  const workerConstructor: new () => Worker;
  export default workerConstructor;
}
