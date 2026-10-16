const COPY: Record<string, { title: string; job: string; lang: string }> = {
  league: { title: "League", job: "Survey the whole league, and test what-ifs.", lang: "Small-multiples atlas of every team, plus the what-if console." },
  trades: { title: "Trades", job: "Negotiate: see a deal from both sides.", lang: "Deal desk with your view and the view their app shows them. Live trade math is already in the engine (~0.1s per deal)." },
  journal: { title: "Journal", job: "Commit: record a decision, why, and how sure you were.", lang: "Field-notes memo with margin notes, scored against outcomes later." },
};

export function Placeholder({ screen }: { screen: string }) {
  const c = COPY[screen];
  return (
    <div style={{ padding: "48px 40px", maxWidth: 640, display: "grid", gap: 10 }}>
      <h2 style={{ margin: 0, font: "700 24px/1.2 var(--sys)" }}>{c.title}</h2>
      <p style={{ margin: 0 }}>{c.job}</p>
      <p className="dim" style={{ margin: 0 }}>{c.lang}</p>
      <p className="dim" style={{ margin: 0 }}>Coming after the My Team check-in.</p>
    </div>
  );
}
