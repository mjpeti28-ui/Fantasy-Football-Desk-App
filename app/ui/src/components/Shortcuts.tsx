import { useEffect } from "react";
import "./palette.css";

/* "?" — every shortcut on one sheet, in the same Spotlight idiom as ⌘K. */
const KEYS: [string, string][] = [
  ["⌘K", "Find a player, team or command"],
  ["⌘1 … ⌘9", "My Team, League, Trades, Waivers, Journal, Shifts, Live, Matchups, Lab"],
  ["Space", "Open the selected player's file"],
  ["← →", "Previous / next player in an open file"],
  ["⌘-click a player", "Add him to Compare (up to 4)"],
  ["Double-click a player", "Open his file"],
  ["⇧⌘C", "Copy this week's lineup plan for Claude"],
  ["⇧⌘F", "Full screen"],
  ["Esc", "Close the file, Compare, or this sheet"],
  ["?", "This sheet"],
];

export function Shortcuts({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === "Escape" || e.key === "?") { e.preventDefault(); onClose(); } };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <div className="pal-scrim" onMouseDown={onClose}>
      <div className="pal keys" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" onMouseDown={(e) => e.stopPropagation()}>
        <h2>Shortcuts</h2>
        <dl>{KEYS.map(([k, v]) => <div key={k}><dt><kbd>{k}</kbd></dt><dd>{v}</dd></div>)}</dl>
      </div>
    </div>
  );
}
