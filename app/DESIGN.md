# Desk — design system (2026-10-07)

Each screen gets its own visual language, chosen for the job it does. A thin shared layer
keeps them one app. Mockups: artifacts "Desk Design Directions" (A–D) and "… II" (E–I).

## Shared layer (every screen)
- **Shell = A, Mac utility.** Native title bar with a segmented screen switcher, source-list
  sidebar (teams, saved items), content area, and an inspector on the right that explains the
  current selection. System font (SF) for all chrome. Follows macOS light/dark.
- **One type family, tuned per screen.** Archivo (variable width 62–125, weight 400–900) is the
  content face everywhere; each screen picks its width. Literata only for long-form writing
  (journal, memo). Overpass Mono only for clocks and console readouts. Nothing else.
- **Color roles, never reused for anything else:**
  - `you` cobalt `#1d4ed8` (your team, your pick, your side of a trade). Matches the system
    selection color so the shell and content agree.
  - `them` plum `#8b3f86` (the opponent or trade partner).
  - `above` green / `below` red: beat or missed a projection, gained or lost.
  - `alert` red hatch: bye, out, IR, something that needs you.
  - `noise` grey, dashed: an edge too small to trust (<10 trade points).
  - Position colors (10/7, replaced the muted set, which failed colorblind checks): the dataviz
    reference categorical slots in a validated order — QB orange, RB aqua, WR violet, TE yellow,
    K magenta, DST green (light and dark steps in base.css; all CVD/normal-vision gates pass for
    adjacent pairs). Used for position bands, chips and the points-by-position bars.
- **Shared components:** player face (ESPN headshot, rounded rect), position band,
  number with range-on-hover, status mark (Active / Q / Out / IR / Bye), confidence label
  (noise / thin / % right), data-age stamp, "Copy for Claude" (⌘⇧C, Markdown).
- **Numbers:** tabular figures always; one decimal for points, whole percents, "−" for minus.

## Screen languages
| Screen | Job | Language (from mockup) | Archivo width |
|---|---|---|---|
| My Team | arrange | B magnet board: tiles, bench tray, hatched byes, bye grid | 72 (condensed, athletic) |
| My Team › Game day | time | E timetable: kickoff columns, running total, late swaps | 100 + Overpass Mono clocks |
| Compare (any ⌘-click / "Compare…") | weigh | C spec sheet: options × evidence, best per row filled | 100 body, 66 headline |
| League | survey | G atlas: small multiples, one scale, figure captions | 85 |
| League › What if | test | F console: switches, readouts, ranked sensitivities | 100 + Overpass Mono readouts |
| Trades | negotiate | H deal desk: two-sided tug bars, your view and theirs | 110 (wide, firm) |
| Shifts | spot | L change map: Δsnap × Δshare scatter cut into the trend-tag zones, each labelled with how often changes like it held (2016–25); dumbbells (from → to) below | 90 |
| Waivers | pick up | J scouting ladder: per-position depth strips on one ppg scale (yours filled cobalt, best free agents hollow, replacement level dashed), then pickups ranked by season gain with the best drop | 92 |
| Journal | commit | D field notes: memo, your words in cobalt ink, the model's numbers and the outcome in the margin, calibration plot on top | Literata |
| Lab (⌘9) | explore | P instrument bench: any stat × any stat scatter, this season's fit and r, outliers named, each axis with its backtest chip | 100 + Overpass Mono readouts |
| Menu bar | notice | I native menu-bar extra, notifications, palette | system |

## Edits made for cohesion (vs the mockups)
- "You" is the same cobalt everywhere (mockups used amber, green, teal and blue).
- The opponent/partner is always plum (timetable used grey, deal desk plum).
- Board, atlas and timetable fonts collapse into Archivo at different widths.
- Spec sheet's "best in row" fill uses ink, not the accent, so cobalt still only means "you".

## League analysis (10/7)
Below the table: positional strength matrix (diverging above/below, value printed, hatch = streamer
starts there 30%+ of weeks) with trade fits; seed-finish strips (ink ramp, you cobalt, opponent
plum); strength-vs-luck scatter (only the x axis has carried forward historically); season-ahead
calendar (more fill = stronger week, your softest week outlined); points-by-position 100% bars.
All simulation-based panels come from the same common-random-numbers run as the odds.

## Waivers and Journal notes
- Waivers: season gain = simulated rest-of-season lineup points after the add and the best drop
  (200 sims, same seed for every candidate). Under ±10 is noise and drawn grey. The headline names
  the one move that clears noise and says why (e.g. one healthy QB → bye/injury cover).
- Waivers, "Won't drop": players you won't let go are saved by the engine (app/cache.nosync/prefs/
  waiver_keep.json, shared by every window and the warm-up) and are never offered as the drop; the
  search then tries the next-cheapest players plus everyone at the candidate's position. Set from the
  "Keep <usual drop>" button, the chip row, the "Choose who you'd drop…" checklist, or "Don't drop X"
  in the inspector. Their drop cost still shows; the board rebuilds in a few seconds.
- Journal: one shared "Log to journal" sheet (My Team inspector, Trades, Waivers). Lineup calls
  are scored after the week; waiver/trade calls keep a running in-vs-out point total until week 17.
  Only entries marked "Did it" count toward calibration; "Didn't" entries still show what would
  have happened. Stored in history/journal/<season>.jsonl.

## Roles (usage tags), model v2
- Descriptive only; never fed into valuation. Backtests in app/engine/roles_fit.py (prints everything).
- Data: roles_data.py builds one row per player-game from nflverse play-by-play, FTN charting, snap
  counts and weekly box scores (pandas over HTTPS; cached history/nflverse/<season>/roles_weekly.csv).
  The model is fit on 2016-25 (10 seasons).
- Stabilization: split-half (odd/even games) reliability per metric gives M = games until half the
  spread between players is real. Volume (snap %, shares) M ≈ 1 game; aDOT ≈ 4; deep rate ≈ 9;
  TDs/game 7-17; YAC/catch 15-24; catch rate 17-27; yards per carry ≈ 30, yards per target ≈ 32-36.
  Every in-season number is shrunk toward last season (regressed by its year-to-year r) or the
  position average by R(n) = n/(n+M). Percentiles, traits and the role map use the shrunk value;
  traits are only shown when R ≥ 0.45.
- Styles: 3 groups per position on reliable shape features (RB target mix + passing-down tilt;
  WR aDOT, behind-the-line and deep rate; TE targets/snap, aDOT, behind-the-line; QB designed runs,
  scramble rate, aDOT). Red-zone ratios and contested rate were dropped (reliability ≈ 0.1-0.3).
  Classification widens each group's covariance by the n-game sampling noise: at 3 games the label
  matches the full-season call 63-78% of the time and the stated confidence matches that (old
  version claimed ~88% and was right ~67%). Year-to-year same style 65-81%.
- WR/TE styles with PFF alignment (model v3, 10/16, Max's call): WR and TE add PFF receiving/summary
  alignment (share of pass snaps in the slot / inline, routes per pass snap), fit on 2021-25 weeks 1-12
  (`python -m app.engine.roles_fit --pff <roles cache> <pff cache>`). This reverses the earlier "No PFF
  data" stance for Roles only; Matchups stays free-data. WR is 2 groups (Slot receiver 45% / Outside
  receiver 55%), TE 2 groups (Inline TE 30% / Move TE 70%); RB and QB unchanged. Same rows, shipped
  noise-aware classifier, vs the free styles: WR label at 3 games matches weeks 1-12 85% (free 3-group
  61%), year-to-year same style 78% (67%; kappa .54 vs .41). TE 82% (65%) and 77% (69%; kappa .38 vs .34).
  Fewer groups make the raw rates easier, so read the kappas. WR at 3 groups (BIC's pick: slot / outside
  primary / outside vertical) labelled earlier but was no steadier year to year (kappa .41 vs .41), so
  the vertical split is left to the "Deep aDOT" trait. PFF alignment settles fast (slot rate M ≈ 0.4
  games, TE inline 0.6, route rate 2).
  Live: roles_pff.py pulls this season to date (one API call, at most every 12 h, from the Refresh
  steps; key PFF_API_KEY from .env.local, never logged) into .cache/pff/ (gitignored; PFF rows are never
  committed: roles_model.json holds only model parameters and aggregates). No key, no network or no PFF
  match for a player: that player is typed with the free-data styles (model["style_free"]), and the role
  map switches to the free axes when the season has no PFF data. The role map's PFF axes are season to
  date only, so the player's point has no weekly path.
- What styles do not do: at the same points per game, weekly volatility (CV) and TD share barely
  differ by style; game-script sensitivity vs ESPN's projection is small (+0.02-0.08 pts per point
  of spread). So styles describe, they don't forecast.
- Trends: snap jumps hold 59-65% (73-78% while the absent teammate stays out); snap drops stick
  41-44%; share spikes without more snaps 39-47%. Versus ESPN's next-week projection (2021-25),
  tagged players score about as projected, except RB snap drops (-0.8 pts, ESPN too slow) and TE
  target-share drops (+0.9, ESPN overreacts).

## Player file (K — card back)
- Opens over any screen: double-click a tile/row (My Team board and game day, Waivers table, Trades
  rosters), "Open player file" in the side panel, a player name in League › What if or a Journal
  entry, or press space on the selected player. ←/→ step through the list it came from; esc closes.
- Sections: header in the team's matte colors with four tiles (ESPN this week, model rest of season,
  season so far with rank, comps median); the read (plain sentences tagged Who / Real / Noise /
  Changed / Outlook, each jumps to its evidence); role map (every player at the position on the two
  style axes, group ovals, his weekly path and last season; click a dot to open that player);
  fingerprint (raw vs best guess on a percentile track, signal %, rank this season); by week (usage
  share bars + snap line; points bars green/red vs ESPN's projection tick); similar players
  (40 nearest player-seasons at the same week 2016-25, beeswarm of what they scored after, middle
  50/80% band, kept-role / availability / weekly 15+ and <5 rates, table with week-by-week sparks);
  box score.
- Comps backtest (leave-one-season-out, weeks 3/5/8/11): 10-90% range held the outcome 77-81% of the
  time; median error equals a simple regression on points so far + last season (RB 2.9, WR 2.7,
  TE 1.9, QB 3.6 PPR/g) and beats points so far alone. The UI says the range is the honest part.

## Compare (C) and Shifts (L), 10/8
- Compare: ⌘-click any player (My Team tiles, Waivers/Trades rows), "Compare…" in the side panel,
  the checkbox on a Shifts row, or "Compare with…" in a player file. A floating tray holds up to 4
  (oldest drops off). The question picks the sheet: "Who to start this week" leads with a
  head-to-head split bar (2) or a pairwise matrix (3–4) from ESPN's projection × the position's
  actual/forecast spread; "Who to hold or get" leads with overlapping comps ranges (50/80% bands,
  median tick, model ▲) and, for 3–4, the comps-pairing matrix. Rows fill the better cell only when
  the gap beats that row's noise (projection 1 pt, model 0.75 ppg, percentiles 10, rates 5–8);
  otherwise the label says "≈ toss-up". Byes/out players are named in the verdict and dropped from
  the odds. When our model and the comps disagree, the verdict says so instead of picking.
- Pairwise odds treat players as independent (same-game correlation ignored). Comps pairing is the
  share of the 40×40 comp outcomes where one is ahead, not a forecast of the gap.
- Shifts: league-wide snap/share changes (last 2 games vs the 2 before, or this season vs last).
  Filters: window, position, who has them (everyone / free agents / yours / other teams), direction,
  usually holds (hold rate ≥ 55%), teammate out, hide out/IR, search. Dots: yours cobalt, free
  agents hollow in position color, others grey; size = model ppg; click a zone to filter. QBs have
  no share axis, so they only appear on the map under the QB filter. Zone rectangles are guides at
  the trend-tag thresholds (±15 snap, ±5 share, +10 snap for "both up"), not a model of their own.
- Engine: `GET /api/shifts` (shifts.py, read-only over roles tables, rosters, ledger and ESPN week).
- Dev note: iCloud evicts files in this folder (history/nflverse/2026/roles_weekly.csv was evicted
  within a day). Native reads re-download them; reads from a Linux VM fail with EDEADLK.

## Speed, iCloud, and the screen pass (10/8, later)
- iCloud (Optimize Mac Storage) had evicted ~19,000 of ~19,600 files in .venv, nearly all of
  node_modules, fresh .pyc files and the journal; every launch re-downloaded pandas/numpy.
  app/run.sh now keeps rebuildable things out of iCloud: Python env in ~/.local/share/desk/venv
  (.venv links to it; old one kept as .venv.icloud-old), Rust build in ~/.local/share/desk/target,
  .pyc in ~/.local/share/desk/pycache, UI packages in app/ui/node_modules.nosync (node_modules
  links to it), Desk caches in app/cache.nosync. It also reads the evicted model inputs in the
  background at launch so they download in parallel. One-time setup: pip + npm + a Rust rebuild.
- Engine warm-up: on start (and after Refresh) a thread precomputes inputs → model context →
  waivers → roles → shifts → league odds → your player files; GET /api/warm reports progress
  (titlebar shows "Precomputing…", sidebar dots mark screens still loading). Waivers and league
  odds are saved to app/cache.nosync keyed by input file times + week + code version, so a
  relaunch on unchanged data is instant (VM: 41 s cold → 7 s, then ~2 s).
- Waivers tries each candidate against the 3 cheapest drops + everyone at his position (same
  best drop as trying all 14 for all 36 candidates in week 5, a third of the time), and now
  returns what each rostered player carries (season lineup points lost without him).
- ⌘K palette (system font, Spotlight idiom): any player → file (⌘↵ compare), teams → League or
  a trade desk, screens and actions. Ownership is only shown when rosters/shifts know it.
- My Team board: "Slot by slot" (tug of war per lineup slot vs the opponent, follows the plan
  live, hover = head-to-head odds) and "What each player carries" (bars, hatched band = under
  5 points = noise; starting-but-covered players called out as cheapest to trade away).
- Game day: Vegas game totals per window, implied team total per player, notes for games both
  lineups share (who a shootout helps; defenses move the other way) and same-offense stacks.
- League: "The points race" first in Analysis (cumulative points vs league average, played
  solid, simulated average dashed, ±23/week luck note), because seeding is by total points.
- Trades empty desk: "Who to call" sheet ordered by roster fit × how each manager has answered
  your offers; their "spare part" = second-best healthy player where they're deep.
- Waivers: "Usage jumps on the wire" cards from Shifts (snap jumps / both-up / target spikes,
  with hold rates and the ESPN-already-prices-it caveat); usual drop and what he carries.
- Journal: "Worth logging this week": closest start/sit calls, the best waiver move, the
  biggest role change on your roster as a hunch; each opens the log sheet prefilled.

## Out of iCloud, a launcher, and What's new (10/8, evening)
- The `.nosync` trick failed for node_modules (npm replaces a linked node_modules with a real
  folder), so the repo itself moves: app/relocate.sh copies it to ~/Developer/<name>, reading
  every file natively so evicted ones download, checks every file's size against the original,
  renames the old folder "<name> (old copy in iCloud)" and leaves a small real marker folder
  (MOVED.txt) at the old path. It runs
  once from app/run.sh (or double-click app/move-out-of-icloud.command). Opt out with
  DESK_NO_RELOCATE=1; another home with DESK_HOME=…
- Done on Max's Mac (10/8): the repo now lives at ~/Developer/PFF-fantasy-ranking; the old
  folder is "PFF-fantasy-ranking (old copy in iCloud)" next to a marker folder at the old Desktop path (safe
  to delete the old copy once things look right, along with .venv.icloud-old,
  app/ui/node_modules.icloud-old and the old Rust target folder in it). The script is resumable: an
  interrupted copy is picked up by the next run (marker file .desk-relocating), and it first
  reads the folder 32 files at a time, because iCloud downloads about a file a second one by one.
  Git's index shows tracked .DS_Store files as deleted (they aren't copied); that's expected.
- ~/Applications/Desk.app (installed by app/install-launcher.sh, called from run.sh): an agent
  app whose executable runs app/run.sh inside your login shell (so nvm's node and cargo are on
  PATH) and logs to ~/Library/Logs/Desk.log. Open Desk from Spotlight/Launchpad.
- What's new (titlebar): the engine saves a compact league state per pull in
  app/cache.nosync/states (owners, injury tags, this week's ESPN projection, our ROS ppg; first
  baseline = the weekly run's espn_live.csv). GET /api/changes?since=<state> diffs your team,
  this week's opponent (tags, projection moves ≥1.5) and league adds/drops/trades. Green/red =
  good/bad for you. Opening the panel marks the newest pull as seen.
- Automatic quick refreshes (rosters + projections, no play-by-play): every 8 h, every 3 h on
  a game day, every 25 min in the 3 h before any of your or your opponent's kickoffs.
- My Team lineup check (above the lineup): ESPN's best lineup when your plan isn't it, with the
  head-to-head odds for a single swap and a "Use it" that changes as few slots as possible; and
  "To do in ESPN", the slot moves that turn ESPN's lineup into your plan (copyable).
- "?" opens a shortcuts sheet; Desk reopens on the screen you last used.


## M — Live (10/8, built during Bucs–Cowboys TNF)
- Job: follow the matchup while games are on. Broadcast idiom: a scorebug (points so far, live win
  chance with the change since kickoff, where each side is headed with a p10–p90 range), a strip
  of the games your starters or theirs are in (clock, down and distance, red zone in alert red), the
  win chance through the night (stepped line; big dots = a starter scoring, small = score changes and
  quarter ends; dead time between game windows squeezed out), every starter as a bullet (bar = points
  now, dashed tick = projection pro-rated to the game clock, grey band = likely finish, ring = the
  model's finish), and one player's game as a timeline (points vs an even-pace diagonal, every look
  along the bottom: hollow = incomplete, red ring = inside the 20). Feed: usage vs his usual, target
  share vs season, red-zone looks, quiet stretches, injury mentions in play text, game script, hot
  starts. ⌘7; a red dot in the sidebar while a game with a starter is on.
- Engine: app/engine/live.py, GET /api/live. ESPN's public site API (scoreboard, summary box
  score) + core API play-by-play (participants with athlete ids). server.py reloads live.py when the
  file changes, so it can be changed while Desk runs. Polls every 12 s on the Live screen during
  games, every 60 s elsewhere while a game is on, every 5 min otherwise.
- Model: rest of game = ESPN projection × time left × usage tilt, spread shrinking with sqrt(time
  left); P(win) by 20k-draw simulation over the position's actual/forecast ratio quantiles (players
  independent). Checked on 2025 play-by-play (1,939 WR/TE/RB games): sqrt rule matched the spread
  at every quarter; usage so far is worth ~4 games of season average (fitted); points so far add
  nothing beyond usage (corr 0.03). D/ST tiers apply to the simulated final score, not banked.
- Approximate: D/ST and K use ESPN default scoring from memory (league is private; the public feed
  can't read its settings). Two-point tries come from play text. Replaying week 4 through it
  matched ESPN for Lamb (41.3) and Allen (18.52).
- tauri: the window now stops any old engine before starting its own, so a relaunch always gets
  current engine routes.
- run.sh/relocate.sh (10/8, late): a Terminal tab left in the old folder kept running the renamed
  old copy, and relocate.sh copied it into ~/Developer again each time (four duplicate folders
  "PFF-fantasy-ranking (old copy in iCloud)…"). Now run.sh hands off to the real repo from any
  folder with MOVED.txt or "(old copy in iCloud)" in its name, and records the real repo in
  ~/.local/share/desk/repo; relocate.sh refuses to move an old copy.
- Link vs folder at the old path (10/8, late): relocate.sh first left a symlink at the old Desktop
  path. Claude's connected-folder mount fails on a symlink ("failed to mount"), so every chat
  attached to that path lost its shell. Fixed on disk: that path is now a real folder holding only
  MOVED.txt (it mounts again and says where the repo went), the four duplicate copies in
  ~/Developer and their dangling links were deleted, and the original iCloud folder is back to the
  single name "PFF-fantasy-ranking (old copy in iCloud)". relocate.sh now leaves that marker
  folder instead of a link, and run.sh finds the new repo through ~/.local/share/desk/relocated-to.
  The app's saved folder for a chat can't be changed from outside: attach
  ~/Developer/PFF-fantasy-ranking there to work on the repo.

## Driving the real window, and the 10/9 polish pass
- `tauri dev` runs a bare binary that computer use (and Spotlight) can't list. `app/build-app.sh`
  builds a real bundle (`com.maxpetite.desk`, `tauri build --debug --bundles app`, target under
  ~/.local/share/desk/target) and installs it as `~/Applications/Desk (built).app`. The built app
  embeds the UI at build time (`frontendDist ../dist`), so after a UI change the built app shows
  nothing new until `app/build-app.sh` runs again; `app/run.sh` (dev) always shows the source.
- Found by using the real window: League's Playoffs column, the Trades call sheet and the Waivers
  and Shifts tables clipped at the default width. They now reflow (the call sheet drops its last
  two columns under 1560 px) and tables scroll sideways inside their panel, never the page. The
  window's minimum width is 1280. The Chromium audit (`winfit`) finds no overflow at 1440x889 and
  1280x800, but WebKit text is wider than Chromium's, so leave slack when sizing columns.
- One scroll container serves every screen, so each screen opened wherever the last was scrolled.
  Scroll position is now remembered per screen (My Team board and Game day separately).
- Palette: a query matches initials, or letters that start at a word boundary, from 3 characters.
  Scattered-letter hits ("stin" listing Austin Hooper) are gone.
- Live: Copy for Claude lives in the header like every other screen. The win-chance chart's y axis
  zooms to the range the line occupies (the caption says so), because a close game drew a flat
  line on 0-100%.
- Game day: each kickoff window is tagged Live or Final; once a player's game has started the
  number is his actual points and the projection moves to the subline. Started and finished state
  comes from the Live feed (same week only), so it follows `/api/live` polling.
- Inspector: the weekly sparkline has per-week tooltips; the team inspector prints the numbers.
- Process: check that a deploy landed (md5 on the Mac vs the container copy, or `git show --stat`)
  before committing. adb9b67 was an empty commit for want of that check.

## N — Matchups (10/10)
- Job: understand how a player has been used, which coverages and defenses he does and doesn't beat,
  and how each defense (scheme, what it allows, its defenders) shapes that. ⌘8, no inspector.
- Language: scouting room. A slate on the left (your players, their starters, a lookup for anyone),
  a versus band, a plain-words read, and the player and the defense side by side. Bars are "tapes":
  fill from zero, a tick for the reference (position norm, league, his own overall rate), a value,
  and the sample beside it. Stacked bars for mixes that sum to 100% (depth, location, coverage
  shell, drawn in an ink ramp because the categories carry no good/bad). A centered tape with a ±1 s.e.
  whisker for quarterbacks' EPA. A man-rate line chart per defense against the league's own number
  each season. Cobalt = your player, plum = theirs, ink = the defense. No green/red: a defense isn't
  good or bad for you until you know which side of it you're on, and a nudge under the noise is grey.
  Dashed outline = resting on too little (<30 targets, <40 targets allowed, <6 games).
- Views: Week N (slate → matchup) and All defenses (sortable table: expected man rate, blitz,
  pressure, a pass and a run rating (see N2), and which of your slate faces each one; click for the file). Start/sit is the third view (N2).
  "Look through" runs the same player against any other defense. Copy for Claude on both.
- Honesty rules (kept in the numbers, not just the copy): coverage splits are 2023–25 because the
  coverage file for the current season is published after it ends; each defense's expected 2026 mix is
  last season's distance from its own season's league times the measured year-to-year carry-over
  (the league's charted man rate moved 41% → 48% → 31%, so teams are only read against their own
  year; a pooled baseline once made every defense look light on man coverage, tests now pin this).
  Target-rate splits by coverage are shrunk toward the position norm by empirical Bayes; man−zone
  target rate has ~50% signal, PPR per target ~0%, so efficiency splits are printed grey/dashed.
  The net nudge (targets and PPR vs an average defense) is compared with the position's
  game-to-game swing (WR ±7.7); it is always labelled below noise when it is, which matches the
  opponent-defense backtest (backtest/opp_defense_signal.py: null). No projection edge is claimed.
- Defenders: the free data has no coverage assignments, so there is no per-defender coverage grade.
  What it has: pass rush and tackle events (pbp), snap share and team EPA allowed with and without
  each defender (last season's participation, ± two standard errors, flagged when he has since left).
- Data and licences: nflverse play-by-play + FTN charting and participation (CC BY-SA 4.0, credit
  "FTN Data via nflverse" is printed on the screen). No PFF data here (their terms bar feeding it to
  LLMs; Roles WR/TE alignment is the one exception, by Max's choice 10/16) and nothing from Pro Football Reference (their policy bars AI use).
- Engine: app/engine/matchups_data.py (build + cache per season under cache.nosync/matchups; the
  current season refreshes when older than 6 h, offline falls back to the stale file),
  matchups.py (tables, defense, player, defenders, view), GET /api/matchups, /defense/{abbr},
  /player/{espn_id}?opp=. Warm step "Matchup tables". v2 modules and weekly_run.py untouched.
  Tests: tests/test_matchups.py.

## N2 — Matchups: the start/sit card and duel (10/10, later)
- Ask: make Matchups usable for start/sit even though it cannot project. Chosen: a card on every
  player-vs-defense view, and a duel that puts two or three candidates for one slot side by side.
- What the backtests said (2023–25, weeks 4 on, against ESPN's pre-kickoff projection;
  backtest/matchup_mechanisms.py writes app/engine/matchups_credit.json, which the engine reads):
  - A defense's rating (EPA allowed per play, this season so far blended with last season, prior of
    250 plays, carry-over 0.35, centred on the league) does predict the next game's offense EPA per
    dropback: slope +0.57 [0.31, 0.87] over 1,342 games. The matchup is real.
  - ESPN's projection already moves with it. Per +1 sd of a softer pass defense: QB +1.0 pts
    [0.6, 1.4], RB +0.4 [0.2, 0.6] (run defense +0.2 [0.0, 0.4]), WR +0.2 [0.1, 0.4], TE +0.3 [0.1, 0.5].
  - After ESPN's number every left-over interval includes zero (QB -0.4 [-1.0, 0.2], RB pass +0.2
    [-0.2, 0.6], RB run -0.1 [-0.5, 0.4], WR -0.15 [-0.5, 0.2], TE +0.1 [-0.4, 0.6] per sd). So the
    rating explains the number; it adds nothing to it. No difference in spread by defense tercile.
  - The coverage lens (man/zone trait x the defense's expected man rate) is calibrated when the
    game's real coverage is known (slope 0.97 [0.54, 1.42], 2024-25 out of sample), but a defense's
    man rate is predictable to about 0.05 targets a game against a +-2.3 target miss. It is shown
    as "Too small", never as an input.
- The card (top of the week view): ESPN's number large and untouched; a band of +-1 typical miss
  with a mark for how much of the number is the matchup (usually a dot near the centre, which is
  the point); rows for pass defense, run defense (backs), coverage mix, pass rush (QBs) and role,
  each tagged Priced in / Too small / Context / Role with the evidence in the tooltip; a headline
  and three short lines. The headline is "Go with ESPN's X. The matchup is already in it." when he
  plays, and says so plainly when he is out or on a bye. It never edits the projection.
- Role size (10/10, later): the card's "worth X points" is a share of the player's own ESPN projection, not one number per position
  (a 16-point back has more to move than a 6-point one). ESPN's own movement fits the share model as well as the flat one; per +1 sd of a softer
  defense: QB pass 5.5% [3.4, 7.5], RB pass 3.4% [2.1, 4.9], RB run 1.3% [-0.3, 2.8], WR 1.6% [0.3, 2.9], TE 2.6% [0.4, 4.8]. By role, ESPN moves a
  lead back +0.48 pts per sd on pass and run defense, a change-of-pace back 0.00 and -0.16 (matchups_credit.json `tier_credit`, noisy). Part of a
  ten-season study (project doc role-defense-split-2026-10.md): the same defense moves a bigger role about the same percent of its average.
- One style row: a back tagged Early-down runner gets a Context row saying run defense matters about 4x more for that style (+0.86 pts per sd vs
  +0.26 and +0.15; all three eras, 2016-25; backtest/role_defense_holdout.py export writes the numbers). It explains the number and changes nothing.
- "Look through" is a what-if: the card is dashed, labels ESPN's number as his real game, and says
  what the other defense would be worth inside a number like that ("not a forecast").
- The duel (Start/sit view): slot chips for your starters, candidate chips (up to three, any
  eligible player or anyone from search), a table of ESPN, model, defense grade, matchup in the
  number, coverage mix, role, big-week and dud odds (cut-offs by slot: QB 20/10, TE 12/4, otherwise
  15/5), odds he outscores the others (exact over the 2021-25 actual/forecast ratio grid, outcomes
  independent), and your win chance with each in the slot. Call tiers: 70%+ a clear start, 58-70%
  a lean, under 58% a toss-up that says to decide on news, not defenses. Opens on your closest call.
- Defenses get a rating on both sides: Tough (rank 1-11), Middle, Soft (22-32), drawn as a label and
  a place in line, in neutral ink (tough or soft is only good or bad once you know whose side you're on).
- Not built, on purpose: variance tie-breaks (no evidence), any adjustment of ESPN's number, a
  defender-out effect (untested; the 2026 participation file does not exist yet, so absences cannot
  be detected from free data).
- Engine: matchups_rating.py (the rating, shared with the backtest), matchups_card.py (grades,
  card, annotate_board), matchups_credit.json (generated). /api/matchups gains pass_grade and
  run_grade per defense; /api/matchups/player/{id} gains `card`. An older running engine sends no
  card and the screen says to reopen Desk. Tests: tests/test_matchups_card.py.


## Analytics layer (10/14): evidence chips, My Team › Analysis
Max asked for more analytics on every screen except Waivers, Journal and Shifts: backtest everything,
keep what's interesting, say how predictive it is. Flags inline, depth in an Analysis tab, plus a Lab.
- **One backtest behind every stat:** `backtest/analytics_metrics.py run` → `app/engine/analytics_registry.json`
  (findings: project doc `analytics-backtest-2026-10.md`). The weekly panel, finish tiers and summaries live in
  `app/engine/analytics.py` and the backtest imports them, so screens show exactly what was tested.
- **Evidence chips** (`components/Analytics.tsx`, `analytics.css`), one per stat, hover for the numbers:
  ink outline + dot = "small edge" (beat the model's own number and held up out of sample; still small);
  grey fill = "in the model" (predicts, already priced); grey outline = "repeats" (a trait, no points);
  grey dashed = "noise" / "weak" / "didn't hold" (the shared noise look). Never cobalt: cobalt means you.
- **Inline flags** (`FlagChip`): hot/cold vs usage (dashed, never green/red, because it fades), snaps
  up/down and model value up/down (green/red: gained or lost, small edge), boom-bust/steady (outline,
  trait). Thresholds: ±3 ppg over usage, ±10 snap-share points last 3 vs season (4+ games), ±1 ppg model
  value over three weeks; boom-bust = top/bottom 15% swing among the position pool.
- **Finish tiers** sized to the league (10 teams): startable = top 10 QB/TE, top 25 RB/WR; smash = top 3 /
  top 6; dud = outside twice the startable pool. Diverging cells: smash solid green, startable light green,
  bench-level neutral, dud light red; the weekly rank is printed in the cell.
- **Percentiles** rank a player among the top 2× league size at QB/TE and 5× at RB/WR by ppg (3+ games),
  on a neutral ink ramp: darker = higher, no good/bad meaning.
- Partial weeks stay out: analytics run through the last week whose NFL games have all finished.
- **My Team › Analysis** (third segment next to Board and Game day, same pegboard and cards):
  scoring vs usage dumbbell (ring = expected points, dot = scored; about 9% of the gap carries), week-by-
  week finish strips with start/smash/dud counts and swing, snap-share small multiples on one 0–100% scale,
  percentile table against the position, your weeks (scored vs best possible vs ESPN's projection, rank
  that week; efficiency and luck are noise), model value moves (small edge). Charts draw at 1:1 width
  (`useWidth`), so text never scales. Copy for Claude copies the analytics table in this mode.
- Inspector: a **Signals** section under the role (flags with chips and their sentences, scored vs usage,
  finish strip, startable weeks, snap share last 3).
- **Player file › Season analytics** (new tab after The read): flags with chips and sentences; three cards
  (scoring vs usage as big numbers; finishes strip + counts + a weekly-spread dot strip with the middle 60%
  shaded and the average marked; last 3 games vs the season for snap share, target share, xFP,
  opportunities and the model's value, with only snap share and model value colored as gains/losses);
  percentile tracks against the position (same track language as the fingerprint). The By week points
  chart gains a ring per game: what his usage usually produces that week.
- **Compare** gains a "Usage vs results, week to week" group (second in "hold or get", after Role in
  "start this week"): usage xFP, scored over usage (never filled: it fades), finishes strip, startable
  weeks, dud weeks (never filled: didn't hold), swing (never filled), snap share last 3, model value
  move. The two small-edge rows only fill when the gap is worth about half a point a game (snap share
  12 points, model value 2.5 ppg).
- Position cells can't claim more than the pooled stat: a per-position "beyond" result that wasn't tested
  out of sample shows the pooled label (WR ppg reads "in the model", not "didn't hold"); a stat that is
  noise at a position (snap trend at TE and QB) says so in its flag.
- **League › Season so far** (below the odds analysis, same almanac cards): every team × every week
  heat table (green/red around that week's league average, depth in standard deviations, score printed,
  a top bar marks the week's high); scored vs best possible (solid = scored, hatched = what the best
  lineup from the same roster adds; noise chip); drafted vs added (share of starting points from a
  team's own picks); draft risers and fallers (positional pick → positional rank in total points). The
  table gains a sortable "Bench left" column. All descriptive: efficiency, luck and weekly swing did not
  repeat in 2021–25, and the chips say so.
- **Trades:** a corner badge on a roster row's face when a player is running hot/cold vs usage (dashed:
  it fades) or his model value moved (solid, green/red); hover for the sentence. On the desk, "Season so
  far" lists each player in the deal: points per game (what a partner's app shows), what his usage
  usually produces, and the model's number (what the gains use). The empty desk gains "Where the season
  and the model disagree": your players scoring 2.5+ ppg above the model and other teams' players 2.5+
  below it (descriptive; check the reason in the player file — it is a perception gap, not an edge).
- **Matchups › Schedule ahead** (fourth mode): your players × weeks next–17, each cell the opponent graded
  by its rating today (pass defense for QB/WR/TE, run defense for RB): soft green, tough red, rank of 32
  printed, byes in the alert hatch, playoff weeks outlined, soft/tough playoff counts. Chip "weak": a
  rating before week W predicted playoff pass defense at r .22 and run defense at .06 (2023–25), and
  points allowed by position did worse (.07). For planning holds and trades, not weekly start/sit.
- **Live › What moved it:** the five biggest single changes in your win chance (scoring plays, score
  changes, quarter ends), each with before → after win % and, when no starter made the play, which
  game it was and whose starters are in it; plus the net by player as a diverging bar (right of center
  helped you), with an "Everything else" row so the bars add up to kickoff → now. Each point in the
  series puts every game back where it stood at that moment. Descriptive. Copy for Claude adds the biggest moves.
- **Lab (⌘9, P — instrument bench):** any stat against any stat for every relevant QB/RB/WR/TE
  (19 stats: usage, results, consistency, trends, the model's number, ECR), filtered by position and by
  everyone / rostered / free agents / yours, colored by owner (you cobalt, this week's opponent plum, other
  rosters grey, free agents hollow) or by position. Hairlines at the medians, this season's least-squares
  line with r, n, slope and the typical distance from the line in a mono readout; the six players
  furthest from the line are named on the plot and listed beside it (click to pin, double-click for the
  file). Presets: luck vs role, season vs model, snaps moving, volume vs swing, startable vs swing. Each
  axis carries its evidence chip; the line and r describe this season only. Copy for Claude exports the
  view and the outliers. Axis choice is remembered.
