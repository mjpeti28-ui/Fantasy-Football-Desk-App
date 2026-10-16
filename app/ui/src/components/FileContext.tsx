import { createContext, useContext } from "react";

/** Open the full player file from anywhere: open(id, list) — list is the order ←/→ step through. */
export const FileCtx = createContext<{ open: (id: number, list?: number[]) => void; compare: (id: number) => void; inCompare: (id: number) => boolean; openCompare: (id?: number) => void }>(
  { open: () => undefined, compare: () => undefined, inCompare: () => false, openCompare: () => undefined });
export const useFile = () => useContext(FileCtx);

/** Small "open file" button for rows that already use click for something else. */
export function FileBtn({ id, list, label }: { id: number; list?: number[]; label?: string }) {
  const { open } = useFile();
  return (
    <button type="button" className="file-btn" aria-label={`Open ${label ?? "player"} file`} title="Open player file (double-click works too)"
      onClick={(e) => { e.stopPropagation(); open(id, list); }} onPointerDown={(e) => e.stopPropagation()}>
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M4.5 1.5h-3v9h9v-3M7 1.5h3.5V5M10.5 1.5 5.5 6.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
    </button>
  );
}

/** "Add to compare" toggle for side panels and files. */
export function CompareBtn({ id, className = "tbtn" }: { id: number; className?: string }) {
  const { compare, inCompare } = useFile();
  const on = inCompare(id);
  return <button type="button" className={`${className} ${on ? "on" : ""}`} aria-pressed={on} onClick={() => compare(id)} title="Compare side by side (⌘-click a player anywhere also adds him)">{on ? "In compare ✓" : "Compare…"}</button>;
}
