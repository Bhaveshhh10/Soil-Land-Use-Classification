"""Dark-theme UI fragments: architecture diagram and navigation cards."""

import streamlit.components.v1 as components

NAV_CARDS = [
    {
        "page": "Predict",
        "title": "Predict",
        "subtitle": "Upload & classify",
        "desc": "Run the agentic router on new satellite imagery with Grad-CAM attention maps.",
        "accent": "#38bdf8",
    },
    {
        "page": "Model Comparison",
        "title": "Models",
        "subtitle": "Benchmark all approaches",
        "desc": "Classical ML, Vanilla MobileNetV2, SE-MobileNetV2, and the full router pipeline.",
        "accent": "#34d399",
    },
    {
        "page": "Training Progress",
        "title": "Training",
        "subtitle": "Curriculum phases",
        "desc": "Validation accuracy across easy (40%), medium (70%), and full-data phases.",
        "accent": "#a78bfa",
    },
    {
        "page": "Explainability",
        "title": "Explain",
        "subtitle": "Grad-CAM & SE blocks",
        "desc": "See where the network attends and how SE channel excitation works.",
        "accent": "#f472b6",
    },
    {
        "page": "About",
        "title": "About",
        "subtitle": "Project overview",
        "desc": "Dataset, architecture details, algorithmic contributions, and references.",
        "accent": "#94a3b8",
    },
]


def render_architecture_diagram(height: int = 780) -> None:
    """Render the pipeline as a self-contained HTML flowchart (not raw markup)."""
    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1419;
    color: #e2e8f0;
    padding: 20px 16px 12px;
  }
  .wrap {
    max-width: 880px;
    margin: 0 auto;
  }
  .title {
    text-align: center;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 18px;
  }
  .flow {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
  }
  .arrow {
    width: 2px;
    height: 22px;
    background: linear-gradient(180deg, #38bdf8, #334155);
    position: relative;
  }
  .arrow::after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 50%;
    transform: translateX(-50%);
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #38bdf8;
  }
  .arrow--fan {
    height: 14px;
    width: 60%;
    max-width: 320px;
    background: none;
    border-top: 2px solid #475569;
    position: relative;
  }
  .arrow--fan::before, .arrow--fan::after {
    content: '';
    position: absolute;
    top: -2px;
    width: 2px;
    height: 14px;
    background: #475569;
  }
  .arrow--fan::before { left: 15%; }
  .arrow--fan::after { right: 15%; }

  .node {
    text-align: center;
    padding: 14px 22px;
    border-radius: 10px;
    border: 1px solid #334155;
    background: #1a2236;
    min-width: 200px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
  }
  .node .tag {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
  }
  .node strong { display: block; font-size: 0.95rem; color: #f8fafc; margin: 2px 0; }
  .node small { display: block; font-size: 0.68rem; color: #64748b; line-height: 1.35; }

  .node--in { border-color: #38bdf8; }
  .node--in .tag { color: #7dd3fc; }
  .node--bb { border-color: #a78bfa; min-width: 260px; }
  .node--bb .tag { color: #c4b5fd; }
  .node--fuse { border-color: #34d399; min-width: 280px; }
  .node--fuse .tag { color: #6ee7b7; }
  .node--cls { border-color: #fbbf24; }
  .node--cls .tag { color: #fcd34d; }

  .grid4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    width: 100%;
    max-width: 720px;
  }
  .tap {
    text-align: center;
    padding: 10px 6px;
    border-radius: 8px;
    background: #1e293b;
    border: 1px solid #475569;
  }
  .tap .ly { font-size: 0.7rem; font-weight: 700; color: #f1f5f9; }
  .tap .dm { font-size: 0.62rem; color: #94a3b8; margin: 2px 0 6px; }
  .tap .se {
    font-size: 0.6rem;
    color: #38bdf8;
    background: rgba(56,189,248,0.12);
    border-radius: 4px;
    padding: 2px 5px;
    display: inline-block;
  }

  .router {
    width: 100%;
    max-width: 520px;
    padding: 16px;
    border-radius: 10px;
    border: 1px dashed #475569;
    background: #121820;
  }
  .router h4 {
    text-align: center;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 12px;
    font-weight: 600;
  }
  .routes {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .route {
    padding: 12px;
    border-radius: 8px;
    text-align: center;
  }
  .route strong { display: block; color: #f1f5f9; font-size: 0.88rem; margin: 4px 0; }
  .route small { font-size: 0.65rem; color: #64748b; }
  .route .cond {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
  }
  .route--hi {
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.35);
  }
  .route--hi .cond { color: #6ee7b7; background: rgba(52,211,153,0.2); }
  .route--lo {
    background: rgba(251,146,60,0.1);
    border: 1px solid rgba(251,146,60,0.35);
  }
  .route--lo .cond { color: #fdba74; background: rgba(251,146,60,0.2); }

  .legend {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 14px;
    margin-top: 18px;
    padding-top: 12px;
    border-top: 1px solid #2a3548;
    font-size: 0.72rem;
    color: #94a3b8;
  }
  .legend span::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }
  .lg-se::before { background: #38bdf8; }
  .lg-cu::before { background: #a78bfa; }
  .lg-ro::before { background: #34d399; }

  @media (max-width: 640px) {
    .grid4 { grid-template-columns: repeat(2, 1fr); }
    .routes { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="title">SE-MobileNetV2 + Agentic Router — inference pipeline</div>
  <div class="flow">

    <div class="node node--in">
      <div class="tag">Input</div>
      <strong>64 × 64 × 3</strong>
      <small>EuroSAT RGB patch</small>
    </div>

    <div class="arrow"></div>

    <div class="node node--bb">
      <div class="tag">Backbone</div>
      <strong>MobileNetV2</strong>
      <small>ImageNet · multi-scale feature taps</small>
    </div>

    <div class="arrow arrow--fan"></div>

    <div class="grid4">
      <div class="tap">
        <div class="ly">block_3</div>
        <div class="dm">16×16×96</div>
        <div class="se">SE → GAP</div>
      </div>
      <div class="tap">
        <div class="ly">block_6</div>
        <div class="dm">8×8×192</div>
        <div class="se">SE → GAP</div>
      </div>
      <div class="tap">
        <div class="ly">block_13</div>
        <div class="dm">4×4×576</div>
        <div class="se">SE → GAP</div>
      </div>
      <div class="tap">
        <div class="ly">out_relu</div>
        <div class="dm">2×2×1280</div>
        <div class="se">SE → GAP</div>
      </div>
    </div>

    <div class="arrow"></div>

    <div class="node node--fuse">
      <div class="tag">Fusion head</div>
      <strong>Concat → 2144-d</strong>
      <small>Dense 512 · BN · Dropout 0.4</small>
      <small>Dense 256 · Dropout 0.3</small>
    </div>

    <div class="arrow"></div>

    <div class="node node--cls">
      <div class="tag">Classifier</div>
      <strong>Softmax · 10 classes</strong>
      <small>Focal loss · 3-phase curriculum</small>
    </div>

    <div class="arrow"></div>

    <div class="router">
      <h4>Agentic confidence router</h4>
      <div class="routes">
        <div class="route route--hi">
          <span class="cond">conf ≥ 0.60</span>
          <strong>SE direct</strong>
          <small>High-confidence prediction</small>
        </div>
        <div class="route route--lo">
          <span class="cond">conf &lt; 0.60</span>
          <strong>KNN fallback</strong>
          <small>k=7 · SE embeddings</small>
        </div>
      </div>
    </div>

  </div>

  <div class="legend">
    <span class="lg-se">SE channel attention</span>
    <span class="lg-cu">Curriculum learning</span>
    <span class="lg-ro">Per-sample routing</span>
  </div>
</div>
</body>
</html>
        """,
        height=height,
        scrolling=False,
    )
