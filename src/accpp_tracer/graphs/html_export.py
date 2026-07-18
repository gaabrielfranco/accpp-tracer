"""Self-contained interactive HTML export for traced circuit graphs.

``circuit_to_html`` renders a ``Tracer`` output graph as a single HTML file
with no external dependencies: token positions on the x-axis, layers on the
y-axis, nodes colored by component kind, query-side (d) vs key-side (s)
edges distinguished by color AND dash pattern, hover tooltips, click-to-pin
highlighting, and a full edge table. Both light and dark color schemes are
supported (OS preference or an explicit ``data-theme`` attribute on the
document root).

Node conventions follow the tracer: internal nodes are 4-tuples
``(layer, head_or_component_label, dest_token_label, src_token_label)``;
root/objective nodes are 2-tuples ``(objective_label, token_label)``.
"""

import json
from pathlib import Path

import networkx as nx

# Component kinds -> (light hex, dark hex), in the validated categorical
# palette order. "AH" and "MLP" dominate real graphs; the rest appear only
# in deeper traces.
_KIND_COLORS = {
    "AH": ("#2a78d6", "#3987e5"),
    "MLP": ("#008300", "#008300"),
    "Embedding": ("#e87ba4", "#d55181"),
    "AH bias": ("#eda100", "#c98500"),
    "AH offset": ("#1baf7a", "#199e70"),
}

_LEGEND_SHAPE_CSS = {
    "AH": "border-radius:50%",
    "MLP": "border-radius:3px",
    "Embedding": "border-radius:2px; transform:rotate(45deg); scale:.85",
    "AH bias": "clip-path:polygon(50% 0, 100% 100%, 0 100%)",
    "AH offset": "clip-path:polygon(0 0, 100% 0, 50% 100%)",
}


def _node_record(n, G):
    if len(n) == 2:  # root / objective node
        return {"id": repr(n), "kind": "root", "label": str(n[0]),
                "token": n[1]}
    layer, comp, dest, src = n
    kind = comp if isinstance(comp, str) else "AH"
    return {"id": repr(n), "kind": kind, "layer": int(layer),
            "head": comp if isinstance(comp, int) else None,
            "dest": dest, "src": src,
            "attn": G.nodes[n].get("attn_weight")}


def circuit_to_html(
    G: nx.MultiDiGraph,
    tokens: list[str],
    n_layers: int,
    out_path: str | Path,
    title: str = "Traced circuit",
    subtitle: str = "",
    stats: dict[str, str] | None = None,
) -> Path:
    """Write an interactive HTML visualization of a traced circuit graph.

    Args:
        G: Graph from ``Tracer.trace`` / ``trace_from_cache`` /
            ``trace_from_probe``.
        tokens: Prompt token labels in position order — must use the same
            labels as the graph's node tuples (i.e. the ``idx_to_token``
            values, with the ``" (n)"`` duplicate suffixes).
        n_layers: Number of model layers (y-axis extent).
        out_path: Destination ``.html`` path (parent dirs created).
        title: Page title / h1.
        subtitle: One-line description under the title (HTML allowed).
        stats: Optional ordered mapping of stat-chip label -> value shown
            under the header, e.g. ``{"P( Mary)": "0.387", "nodes": "63"}``.

    Returns:
        The written path.
    """
    nodes = [_node_record(n, G) for n in G.nodes]
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            "u": repr(u), "v": repr(v),
            "w": round(float(d.get("weight", 0.0)), 6),
            "type": d.get("type"), "svs": d.get("svs_used"),
        })

    kinds_present = [k for k in _KIND_COLORS if any(n["kind"] == k for n in nodes)]
    legend_items = "".join(
        f'<span class="item"><span class="lg-mark" '
        f'style="background:var(--c-{k.replace(" ", "-")}); {_LEGEND_SHAPE_CSS[k]}"></span> '
        f'{("attention head" if k == "AH" else k)}</span>'
        for k in kinds_present
    )
    kind_vars_light = "\n    ".join(
        f'--c-{k.replace(" ", "-")}: {v[0]};' for k, v in _KIND_COLORS.items())
    kind_vars_dark = "\n      ".join(
        f'--c-{k.replace(" ", "-")}: {v[1]};' for k, v in _KIND_COLORS.items())
    chips = "".join(
        f'<span class="chip">{k} <b>{v}</b></span>'
        for k, v in (stats or {}).items()
    )

    data = {"tokens": tokens, "n_layers": n_layers,
            "nodes": nodes, "edges": edges}
    html = (
        _TEMPLATE
        .replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__CHIPS__", chips)
        .replace("__LEGEND_KINDS__", legend_items)
        .replace("__KIND_VARS_LIGHT__", kind_vars_light)
        .replace("__KIND_VARS_DARK__", kind_vars_dark)
        .replace("__DATA_JSON__", json.dumps(data))
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


_TEMPLATE = r"""<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --plane: #f9f9f7;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
    --edge-d: #eb6834; --edge-s: #4a3aa7;
    __KIND_VARS_LIGHT__
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface: #1a1a19; --plane: #0d0d0d;
      --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
      --edge-d: #d95926; --edge-s: #9085e9;
      __KIND_VARS_DARK__
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --plane: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --edge-d: #d95926; --edge-s: #9085e9;
    __KIND_VARS_DARK__
  }
  :root[data-theme="light"] {
    color-scheme: light;
    --surface: #fcfcfb; --plane: #f9f9f7;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
    --edge-d: #eb6834; --edge-s: #4a3aa7;
    __KIND_VARS_LIGHT__
  }

  body {
    margin: 0; background: var(--plane); color: var(--ink);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1560px; margin: 0 auto; padding: 24px 20px 48px; }
  header h1 { font-size: 20px; font-weight: 650; margin: 0 0 2px; text-wrap: balance; }
  header .sub { color: var(--ink-2); font-size: 13px; margin: 0; }
  header .sub .tok { font-weight: 600; color: var(--ink); }

  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 10px; }
  .chip {
    background: var(--surface); border: 1px solid var(--ring); border-radius: 999px;
    padding: 3px 12px; font-size: 12px; color: var(--ink-2); white-space: nowrap;
  }
  .chip b { color: var(--ink); font-weight: 650; font-variant-numeric: tabular-nums; }

  .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center;
            font-size: 12px; color: var(--ink-2); margin: 0 0 12px; }
  .legend .item { display: inline-flex; align-items: center; gap: 6px; }
  .lg-mark { width: 11px; height: 11px; display: inline-block; }
  .lg-line { width: 26px; height: 0; border-top: 2.5px solid var(--edge-d); display: inline-block; }
  .lg-dash { width: 26px; height: 0; border-top: 2.5px dashed var(--edge-s); display: inline-block; }

  .panel {
    background: var(--surface); border: 1px solid var(--ring); border-radius: 10px;
    overflow-x: auto; position: relative;
  }
  svg { display: block; }
  svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }

  .edge { fill: none; opacity: .5; transition: opacity .12s; }
  .edge.s { stroke-dasharray: 5 4; }
  .edge-hit { fill: none; stroke: transparent; stroke-width: 11; cursor: pointer; }
  .node { cursor: pointer; stroke: var(--surface); stroke-width: 2; }
  .nlabel { font-size: 10px; fill: var(--ink-2); pointer-events: none; }
  .dimmed .edge { opacity: .08; }
  .dimmed .edge.hot { opacity: .95; }
  .dimmed .node { opacity: .25; }
  .dimmed .node.hot { opacity: 1; }
  .dimmed .nlabel { opacity: .25; }
  .dimmed .nlabel.hot { opacity: 1; }

  #tip {
    position: fixed; z-index: 10; display: none; max-width: 340px;
    background: var(--surface); color: var(--ink); border: 1px solid var(--ring);
    border-radius: 8px; padding: 8px 11px; font-size: 12px; line-height: 1.45;
    box-shadow: 0 4px 16px rgba(0,0,0,.18); pointer-events: none;
  }
  #tip .t2 { color: var(--ink-2); }
  #tip b { font-variant-numeric: tabular-nums; }

  details { margin-top: 22px; }
  summary { cursor: pointer; font-weight: 650; font-size: 14px; color: var(--ink); }
  .tablewrap { overflow-x: auto; margin-top: 10px; background: var(--surface);
               border: 1px solid var(--ring); border-radius: 10px; }
  table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; letter-spacing: .04em;
       text-transform: uppercase; position: sticky; top: 0; background: var(--surface); }
  td.num { font-variant-numeric: tabular-nums; text-align: right; }
  td .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  tr:last-child td { border-bottom: none; }

  .note { color: var(--muted); font-size: 12px; margin-top: 10px; }
  @media (prefers-reduced-motion: reduce) { .edge, .node { transition: none; } }
</style>

<div class="wrap">
  <header>
    <h1>__TITLE__</h1>
    <p class="sub">__SUBTITLE__</p>
  </header>

  <div class="chips">__CHIPS__</div>

  <div class="legend">
    __LEGEND_KINDS__
    <span class="item"><span class="lg-line"></span> query-side edge (d)</span>
    <span class="item"><span class="lg-dash"></span> key-side edge (s)</span>
    <span class="item">line width &#8733; edge weight</span>
    <span class="item">hover to inspect &#183; click a node to pin</span>
  </div>

  <div class="panel"><svg id="chart" role="img"
    aria-label="Circuit graph: token position horizontally, layer vertically"></svg></div>
  <p class="note">Layout: columns are prompt token positions (a head sits at its <em>destination</em> token);
    rows are layers, layer 0 at the bottom. A query-side (d) edge explains how the upstream component shaped
    what the downstream head was <em>looking for</em>; a key-side (s) edge, what it <em>found</em> at the source token.
    Edge weight is the attention-weight drop when the component&#8217;s singular-vector channels are removed
    (for edges into the root: the seed&#8217;s contribution to the objective).</p>

  <details>
    <summary>All edges (sorted by weight)</summary>
    <div class="tablewrap"><table id="etable">
      <thead><tr><th>From</th><th>To</th><th>Side</th><th style="text-align:right">Weight</th><th>SV channels</th></tr></thead>
      <tbody></tbody>
    </table></div>
  </details>
</div>

<div id="tip"></div>

<script>
const DATA = __DATA_JSON__;

(function () {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("chart");
  const tip = document.getElementById("tip");

  const M = { l: 56, r: 70, t: 64, b: 46 };
  const colW = 92, rowH = 42;
  const nTok = DATA.tokens.length, nLay = DATA.n_layers;
  const W = M.l + nTok * colW + M.r;
  const H = M.t + (nLay + 1) * rowH + M.b;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);

  const tokIdx = {}; DATA.tokens.forEach((t, i) => tokIdx[t] = i);
  const colX = i => M.l + i * colW + colW / 2;
  const layY = l => M.t + (nLay - l) * rowH + rowH / 2;

  function el(name, attrs, parent) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e); return e;
  }
  const disp = s => s === "<|begin_of_text|>" || s === "<|endoftext|>" ? "BOS"
    : (s || "").trim() === "" ? "·" : s;
  const compName = n => n.kind === "AH" ? `L${n.layer}H${n.head}` : `L${n.layer} ${n.kind}`;
  const nodeName = n => n.kind === "root" ? n.label
    : n.kind === "AH" ? `L${n.layer}H${n.head} (${disp(n.dest)} ← ${disp(n.src)})`
    : `${compName(n)} @${disp(n.dest)}`;

  // --- layout: place nodes in (dest-token column, layer row) cells ---
  const cells = {};
  DATA.nodes.forEach(n => {
    if (n.kind === "root") return;
    const key = tokIdx[n.dest] + "," + n.layer;
    (cells[key] = cells[key] || []).push(n);
  });
  const pos = {};
  for (const key in cells) {
    const [ci, li] = key.split(",").map(Number);
    const group = cells[key].sort((a, b) =>
      (a.kind === "AH" ? a.head : -1) - (b.kind === "AH" ? b.head : -1));
    const gap = group.length > 3 ? 19 : 24;
    group.forEach((n, i) => {
      pos[n.id] = { x: colX(ci) + (i - (group.length - 1) / 2) * gap, y: layY(li) };
    });
  }
  // Roots: above their token column; multiple roots on one column fan out.
  const roots = DATA.nodes.filter(n => n.kind === "root");
  const rootsByTok = {};
  roots.forEach(r => (rootsByTok[r.token] = rootsByTok[r.token] || []).push(r));
  for (const t in rootsByTok) {
    const group = rootsByTok[t];
    const cx = t in tokIdx ? colX(tokIdx[t]) : colX(nTok - 1);
    group.forEach((r, i) => {
      pos[r.id] = { x: cx + (i - (group.length - 1) / 2) * 120, y: M.t - 26 };
    });
  }

  // --- chrome: layer gridlines + axis labels + token labels ---
  for (let l = 0; l <= nLay - 1; l += 4) {
    el("line", { x1: M.l - 6, x2: W - M.r, y1: layY(l), y2: layY(l),
                 stroke: "var(--grid)", "stroke-width": 1 });
    el("text", { x: M.l - 10, y: layY(l) + 3.5, "text-anchor": "end",
                 "font-size": 10, fill: "var(--muted)" }).textContent = "L" + l;
  }
  el("text", { x: 14, y: M.t + 10, "font-size": 10, fill: "var(--muted)",
               transform: `rotate(-90 14 ${M.t + 10})`, "text-anchor": "end" })
    .textContent = "layer →";
  const activeCols = new Set(Object.keys(cells).map(k => +k.split(",")[0]));
  DATA.tokens.forEach((t, i) => {
    el("text", { x: colX(i), y: H - M.b + 22, "text-anchor": "middle",
                 "font-size": 11.5, fill: activeCols.has(i) ? "var(--ink)" : "var(--muted)",
                 "font-weight": activeCols.has(i) ? 650 : 400 }).textContent = disp(t);
  });
  el("line", { x1: M.l - 6, x2: W - M.r, y1: H - M.b + 4, y2: H - M.b + 4,
               stroke: "var(--axis)", "stroke-width": 1 });

  // --- edges ---
  const wMax = Math.max(...DATA.edges.map(e => e.w), 1e-9);
  const wScale = w => 1 + 5 * Math.sqrt(w / wMax);
  const edgesG = el("g", {});
  const nodesG = el("g", {});
  const incident = {};
  const byId = {}; DATA.nodes.forEach(n => byId[n.id] = n);

  DATA.edges.forEach(e => {
    const p1 = pos[e.u], p2 = pos[e.v];
    if (!p1 || !p2) return;
    const bow = (p1.x === p2.x) ? (e.type === "d" ? -16 : 16)
                                : (e.type === "d" ? -7 : 7);
    const mx = (p1.x + p2.x) / 2 + bow, my = (p1.y + p2.y) / 2;
    const d = `M ${p1.x} ${p1.y} Q ${mx} ${my} ${p2.x} ${p2.y}`;
    const path = el("path", {
      d, class: "edge " + e.type,
      stroke: e.type === "d" ? "var(--edge-d)" : "var(--edge-s)",
      "stroke-width": wScale(e.w).toFixed(2), "stroke-linecap": "round",
    }, edgesG);
    const hit = el("path", { d, class: "edge-hit" }, edgesG);
    (incident[e.u] = incident[e.u] || []).push(path);
    (incident[e.v] = incident[e.v] || []).push(path);
    hit.addEventListener("pointermove", ev => {
      showTip(ev, `<b>${nodeName(byId[e.u])}</b> → <b>${nodeName(byId[e.v])}</b><br>
        <span class="t2">${e.type === "d" ? "query-side (d)" : "key-side (s)"} ·
        weight <b>${e.w.toFixed(4)}</b>${e.svs ? " · SV channels " + e.svs : " · seed edge"}</span>`);
      path.classList.add("hot");
    });
    hit.addEventListener("pointerleave", () => { hideTip(); path.classList.remove("hot"); });
  });

  // --- nodes ---
  let pinned = null;
  function setFocus(id) {
    svg.classList.toggle("dimmed", !!id);
    svg.querySelectorAll(".hot").forEach(x => x.classList.remove("hot"));
    if (!id) return;
    (incident[id] || []).forEach(p => p.classList.add("hot"));
    const g = nodesG.querySelector(`[data-id="${CSS.escape(id)}"]`);
    if (g) g.querySelectorAll(".node,.nlabel").forEach(x => x.classList.add("hot"));
  }

  function markFor(n, p, g) {
    const v = `var(--c-${n.kind.replace(" ", "-")})`;
    if (n.kind === "MLP")
      return el("rect", { x: p.x - 6, y: p.y - 6, width: 12, height: 12, rx: 3,
                          class: "node", fill: v }, g);
    if (n.kind === "Embedding")
      return el("rect", { x: p.x - 5.5, y: p.y - 5.5, width: 11, height: 11, rx: 2,
                          class: "node", fill: v,
                          transform: `rotate(45 ${p.x} ${p.y})` }, g);
    if (n.kind === "AH bias")
      return el("path", { d: `M ${p.x} ${p.y - 7} L ${p.x + 7} ${p.y + 6} L ${p.x - 7} ${p.y + 6} Z`,
                          class: "node", fill: v }, g);
    if (n.kind === "AH offset")
      return el("path", { d: `M ${p.x - 7} ${p.y - 6} L ${p.x + 7} ${p.y - 6} L ${p.x} ${p.y + 7} Z`,
                          class: "node", fill: v }, g);
    return el("circle", { cx: p.x, cy: p.y, r: 7, class: "node", fill: v }, g);
  }

  DATA.nodes.forEach(n => {
    const p = pos[n.id];
    if (!p) return;
    const g = el("g", { "data-id": n.id }, nodesG);
    let mark;
    if (n.kind === "root") {
      const w = Math.max(72, n.label.length * 6.6 + 20);
      mark = el("rect", { x: p.x - w / 2, y: p.y - 13, width: w, height: 26, rx: 13,
        class: "node", fill: "var(--surface)", stroke: "var(--ink)", "stroke-width": 1.4 }, g);
      el("text", { x: p.x, y: p.y + 4, "text-anchor": "middle", "font-size": 11.5,
        "font-weight": 650, fill: "var(--ink)", "pointer-events": "none" }, g)
        .textContent = n.label;
    } else {
      mark = markFor(n, p, g);
      if (n.kind === "AH") {
        // Label placement by position within the cell: rightmost labels
        // right, leftmost labels left, interior alternates above/below.
        const cell = cells[tokIdx[n.dest] + "," + n.layer];
        const idx = cell.indexOf(n);
        let lx, ly = p.y + 3.5, anchor;
        if (idx === cell.length - 1) { lx = p.x + 11; anchor = "start"; }
        else if (idx === 0) { lx = p.x - 11; anchor = "end"; }
        else {
          lx = p.x; anchor = "middle";
          ly = p.y + (idx % 2 === 1 ? -12 : 20);
        }
        el("text", { x: lx, y: ly, "text-anchor": anchor, class: "nlabel" }, g)
          .textContent = `L${n.layer}H${n.head}`;
      }
    }
    mark.addEventListener("pointermove", ev => {
      let html;
      if (n.kind === "root") {
        html = `<b>Objective: ${n.label}</b><br><span class="t2">read at “${disp(n.token)}”</span>`;
      } else if (n.kind === "AH") {
        html = `<b>Attention head L${n.layer}·H${n.head}</b><br>
          <span class="t2">“${disp(n.dest)}” attends to “${disp(n.src)}”
          ${n.attn != null ? " · attention <b>" + n.attn.toFixed(3) + "</b>" : ""}</span>`;
      } else {
        html = `<b>${n.kind}, layer ${n.layer}</b><br><span class="t2">at token “${disp(n.dest)}”</span>`;
      }
      showTip(ev, html);
      if (!pinned) setFocus(n.id);
    });
    mark.addEventListener("pointerleave", () => { hideTip(); if (!pinned) setFocus(null); });
    mark.addEventListener("click", ev => {
      ev.stopPropagation();
      pinned = (pinned === n.id) ? null : n.id;
      setFocus(pinned);
    });
  });
  svg.addEventListener("click", () => { pinned = null; setFocus(null); });

  function showTip(ev, html) {
    tip.innerHTML = html; tip.style.display = "block";
    const pad = 14, vw = window.innerWidth;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    tip.style.left = "0px"; tip.style.top = "0px";
    const r = tip.getBoundingClientRect();
    if (x + r.width > vw - 8) x = ev.clientX - r.width - pad;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hideTip() { tip.style.display = "none"; }

  // --- table ---
  const tbody = document.querySelector("#etable tbody");
  [...DATA.edges].sort((a, b) => b.w - a.w).forEach(e => {
    const tr = document.createElement("tr");
    const side = e.type === "d"
      ? `<span class="dot" style="background:var(--edge-d)"></span>query (d)`
      : `<span class="dot" style="background:var(--edge-s)"></span>key (s)`;
    tr.innerHTML = `<td>${nodeName(byId[e.u])}</td><td>${nodeName(byId[e.v])}</td>
      <td>${side}</td><td class="num">${e.w.toFixed(4)}</td>
      <td>${e.svs ? e.svs : "—"}</td>`;
    tbody.appendChild(tr);
  });
})();
</script>
"""
