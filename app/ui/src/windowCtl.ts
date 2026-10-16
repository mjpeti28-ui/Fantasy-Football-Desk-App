/** Window full screen through Tauri's window API (no npm package needed: the webview exposes
 *  __TAURI_INTERNALS__). Tries macOS native full screen first; if the window isn't full screen a
 *  moment later, falls back to "simple" full screen (covers the display without a new Space).
 *  In a plain browser it uses the page Fullscreen API. */
type Invoke = (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
const tauri = (): Invoke | null => (window as unknown as { __TAURI_INTERNALS__?: { invoke: Invoke } }).__TAURI_INTERNALS__?.invoke ?? null;
const LABEL = "main";
let simple = false;

export async function toggleFullscreen(): Promise<string> {
  const inv = tauri();
  if (!inv) {
    if (document.fullscreenElement) { await document.exitFullscreen(); return "Left full screen"; }
    await document.documentElement.requestFullscreen?.();
    return "Full screen";
  }
  try {
    if (simple) { await inv("plugin:window|set_simple_fullscreen", { label: LABEL, value: false }); simple = false; return "Left full screen"; }
    if (await inv("plugin:window|is_fullscreen", { label: LABEL })) { await inv("plugin:window|set_fullscreen", { label: LABEL, value: false }); return "Left full screen"; }
    await inv("plugin:window|set_fullscreen", { label: LABEL, value: true });
    await new Promise((r) => setTimeout(r, 900));
    if (await inv("plugin:window|is_fullscreen", { label: LABEL })) return "Full screen";
    await inv("plugin:window|set_simple_fullscreen", { label: LABEL, value: true });
    simple = true;
    return "Full screen (simple mode: native full screen didn't respond)";
  } catch (e) {
    return `Full screen failed: ${e}`;
  }
}
