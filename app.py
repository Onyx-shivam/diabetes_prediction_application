"""
Metabolic profile classifier - Flask app serving svm.pkl (scikit-learn SVC).

Run locally:   python app.py
Run on Render: gunicorn app:app

Keep svm.pkl in the same folder as this file.
"""
import json
import math
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent

# =====================================================================
#  EDIT THIS SECTION to match how you encoded data when training.
#  The pickle stores only the model, not your LabelEncoder / mappings,
#  so each option's number ("value") must match your training notebook.
#  Check with:  encoder.classes_   (index in that list = the number)
# =====================================================================
APP_TITLE = "Metabolic profile classifier"
APP_SUBTITLE = (
    "Enter 17 health details and the SVM model rates the profile "
    "as low, moderate or high risk."
)

# Risk level for each predicted class (0, 1, 2). "rank" orders them from
# lowest (0) to highest (2) risk. Inferred from the model's behaviour:
# class 1 = lowest glucose/HbA1c/BMI profile, class 2 = middle, class 0 = highest.
# Confirm against your training labels and swap the keys here if needed.
CLASS_LABELS = {
    1: {"name": "Low risk", "rank": 0, "color": "#16A34A",
        "note": "This profile resembles the lower-risk cases the model learned from. "
                "Keep up healthy habits and regular check-ups."},
    2: {"name": "Moderate risk", "rank": 1, "color": "#D97706",
        "note": "This profile sits in the middle. Lifestyle changes and a routine "
                "screening discussion with a doctor are worth considering."},
    0: {"name": "High risk", "rank": 2, "color": "#DC2626",
        "note": "This profile resembles the higher-risk cases the model learned from. "
                "Please talk to a doctor soon about testing and a care plan."},
}

# Cities are encoded 0-17 in the model. Replace with your real city names
# in the same order as your encoder's classes_ list (must stay 18 items).
CITIES = [f"City {i}" for i in range(18)]
# Example: CITIES = ["Ahmedabad", "Bengaluru", "Chennai", ...]

# Option lists: (encoded value, label shown to the user)
GENDER = [(0, "Female"), (1, "Male"), (2, "Other")]
YES_NO = [(0, "No"), (1, "Yes")]
ACTIVITY = [(0, "Low"), (1, "Moderate"), (2, "High")]
DIET = [(0, "Type 0"), (1, "Type 1"), (2, "Type 2"), (3, "Type 3")]
ALCOHOL = [(0, "None"), (1, "Light"), (2, "Moderate"), (3, "Heavy")]
INCOME = [(0, "Low"), (1, "Lower-middle"), (2, "Upper-middle"), (3, "High")]
# =====================================================================

# Form definition. "name" must match the model's training columns.
# min/max = hard limits; "train" = range the model's key examples cover.
FIELDS = [
    # ---- About you
    {"name": "age", "label": "Age", "type": "number", "unit": "years",
     "min": 18, "max": 100, "step": 1, "default": 45, "train": [18, 73], "group": "About you"},
    {"name": "gender", "label": "Gender", "type": "seg", "options": GENDER,
     "default": 1, "group": "About you"},
    {"name": "city", "label": "City", "type": "select",
     "options": list(enumerate(CITIES)), "default": 0, "group": "About you"},
    {"name": "income_bracket", "label": "Income bracket", "type": "select",
     "options": INCOME, "default": 1, "group": "About you"},
    {"name": "family_history_diabetes", "label": "Family history of diabetes",
     "type": "seg", "options": YES_NO, "default": 0, "group": "About you"},

    # ---- Lifestyle
    {"name": "physical_activity_level", "label": "Physical activity", "type": "seg",
     "options": ACTIVITY, "default": 1, "group": "Lifestyle"},
    {"name": "diet_type", "label": "Diet type", "type": "select",
     "options": DIET, "default": 1, "group": "Lifestyle"},
    {"name": "smoking_status", "label": "Smoker", "type": "seg",
     "options": YES_NO, "default": 1, "group": "Lifestyle"},
    {"name": "alcohol_consumption", "label": "Alcohol consumption", "type": "select",
     "options": ALCOHOL, "default": 1, "group": "Lifestyle"},
    {"name": "hours_sleep_per_night", "label": "Sleep per night", "type": "range",
     "unit": "hours", "min": 3, "max": 12, "step": 0.5, "default": 7,
     "train": [3, 12], "group": "Lifestyle"},
    {"name": "stress_level", "label": "Stress level", "type": "range",
     "unit": "out of 10", "min": 1, "max": 10, "step": 1, "default": 6,
     "train": [1, 10], "group": "Lifestyle"},

    # ---- Body and lab results
    {"name": "bmi", "label": "BMI", "type": "number", "unit": "kg/m²",
     "min": 10, "max": 70, "step": 0.1, "default": 24.5, "train": [15, 41],
     "group": "Body and lab results"},
    {"name": "waist_circumference_cm", "label": "Waist circumference", "type": "number",
     "unit": "cm", "min": 40, "max": 200, "step": 0.1, "default": 102,
     "train": [73, 149], "group": "Body and lab results"},
    {"name": "fasting_blood_sugar", "label": "Fasting blood sugar", "type": "number",
     "unit": "mg/dL", "min": 40, "max": 600, "step": 1, "default": 185,
     "train": [118, 258], "group": "Body and lab results"},
    {"name": "hba1c_level", "label": "HbA1c", "type": "number", "unit": "%",
     "min": 3, "max": 20, "step": 0.1, "default": 7.2, "train": [5.5, 9.6],
     "group": "Body and lab results"},
    {"name": "blood_pressure_systolic", "label": "Systolic blood pressure",
     "type": "number", "unit": "mmHg", "min": 70, "max": 260, "step": 1,
     "default": 140, "train": [100, 200], "group": "Body and lab results"},
    {"name": "blood_pressure_diastolic", "label": "Diastolic blood pressure",
     "type": "number", "unit": "mmHg", "min": 40, "max": 160, "step": 1,
     "default": 88, "train": [50, 120], "group": "Body and lab results"},
]

# ---------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------
with open(BASE_DIR / "svm.pkl", "rb") as f:
    MODEL = pickle.load(f)

FEATURES = list(MODEL.feature_names_in_)
assert {f["name"] for f in FIELDS} == set(FEATURES), (
    "FIELDS do not match the model's training columns: "
    f"{set(FEATURES) ^ {f['name'] for f in FIELDS}}"
)
FIELD_BY_NAME = {f["name"]: f for f in FIELDS}

app = Flask(__name__)


def parse_input(payload):
    """Validate the request and return (values, errors, warnings)."""
    values, errors, warnings = {}, {}, []
    for name in FEATURES:
        field = FIELD_BY_NAME[name]
        raw = payload.get(name)
        if raw is None or str(raw).strip() == "":
            errors[name] = f"{field['label']} is required."
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            errors[name] = f"{field['label']} must be a number."
            continue
        if not math.isfinite(num):
            errors[name] = f"{field['label']} must be a valid number."
            continue

        if field["type"] in ("seg", "select"):
            allowed = {v for v, _ in field["options"]}
            if int(num) != num or int(num) not in allowed:
                errors[name] = f"Choose a valid option for {field['label'].lower()}."
                continue
        else:
            if not (field["min"] <= num <= field["max"]):
                errors[name] = (
                    f"{field['label']} should be between "
                    f"{field['min']} and {field['max']}."
                )
                continue
            lo, hi = field["train"]
            if not (lo <= num <= hi):
                warnings.append(
                    f"{field['label']} is outside {lo}\u2013{hi}, the range this "
                    "model's key training examples cover, so the result may be "
                    "less reliable."
                )
        values[name] = num
    return values, errors, warnings


def softmax(scores):
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


@app.get("/")
def index():
    config = {
        "title": APP_TITLE,
        "subtitle": APP_SUBTITLE,
        "fields": FIELDS,
        "classes": {str(k): v for k, v in CLASS_LABELS.items()},
    }
    return INDEX_HTML.replace("__CONFIG__", json.dumps(config)).replace(
        "__TITLE__", APP_TITLE
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    values, errors, warnings = parse_input(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    X = pd.DataFrame([values])[FEATURES]
    predicted = int(MODEL.predict(X)[0])
    scores = MODEL.decision_function(X)[0]
    shares = softmax(list(scores))

    classes = [int(c) for c in MODEL.classes_]
    return jsonify({
        "ok": True,
        "predicted": predicted,
        "scores": [
            {"class": c, "share": round(s * 100, 1)}
            for c, s in zip(classes, shares)
        ],
        "warnings": warnings,
    })


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------
# Front end (single page, no build step)
# ---------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#16213E; --muted:#5E6B85; --paper:#EEF1FB; --surface:#FFFFFF;
  --line:#DFE4F3; --soft:#EDF0FA; --accent:#4F46E5; --accent-ink:#FFFFFF;
  --danger:#C23B32; --warn-bg:#FFF4E0; --warn-ink:#7A4A06;
  --shadow:0 12px 32px rgba(40,44,120,.10);
  --radius:16px;
  --display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;
  --body:"Figtree","Segoe UI",system-ui,sans-serif;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);
  font-size:16px;line-height:1.5}
.hero{position:relative;overflow:hidden;color:#fff;
  background:linear-gradient(135deg,#1B1464 0%,#4338CA 55%,#0F8F8B 100%)}
.hero::before{content:"";position:absolute;right:-110px;top:-150px;width:460px;height:460px;
  border-radius:50%;background:radial-gradient(circle at 35% 35%,rgba(255,255,255,.24),rgba(255,255,255,0) 70%)}
.hero::after{content:"";position:absolute;left:38%;bottom:-210px;width:420px;height:420px;
  border-radius:50%;background:radial-gradient(circle,rgba(45,212,191,.35),rgba(45,212,191,0) 70%)}
.hero-in{position:relative;z-index:1;max-width:1120px;margin:0 auto;padding:48px 24px 108px}
.wrap{position:relative;z-index:2;max-width:1120px;margin:-68px auto 0;padding:0 24px 64px}
header{max-width:640px}
h1{font-family:var(--display);font-weight:700;font-size:clamp(30px,4.4vw,46px);
  line-height:1.05;letter-spacing:-.02em;margin:0 0 12px}
header p{margin:0;color:rgba(255,255,255,.84);font-size:17px}

.layout{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:24px;align-items:start}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
form.panel{padding:8px 28px 28px}
.group{padding:24px 0;border-bottom:1px solid var(--line)}
.group:last-of-type{border-bottom:0}
.group h2{display:flex;align-items:center;gap:12px;font-family:var(--display);font-weight:500;font-size:21px;letter-spacing:-.01em;margin:0 0 18px}
.group h2::before{content:"";width:6px;height:24px;border-radius:3px;background:var(--g,var(--accent))}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px 20px}
.field{display:flex;flex-direction:column;gap:7px;min-width:0}
.field>label,.field>.lbl{font-size:14px;font-weight:600}
.hint{font-weight:400;color:var(--muted)}

input[type=number],select{
  width:100%;height:46px;padding:0 14px;border:1px solid var(--line);border-radius:10px;
  background:var(--surface);color:var(--ink);font:inherit;appearance:none;-moz-appearance:textfield}
select{background-image:linear-gradient(45deg,transparent 50%,var(--muted) 50%),
  linear-gradient(135deg,var(--muted) 50%,transparent 50%);
  background-position:calc(100% - 20px) 20px,calc(100% - 15px) 20px;
  background-size:5px 5px;background-repeat:no-repeat;padding-right:36px}
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.input-wrap{position:relative}
.input-wrap input{padding-right:70px}
.unit{position:absolute;right:14px;top:50%;transform:translateY(-50%);
  color:var(--muted);font-size:14px;pointer-events:none}
input:hover,select:hover{border-color:#B8C0E0}
input:focus,select:focus{border-color:var(--accent)}
:focus-visible{outline:3px solid rgba(79,70,229,.35);outline-offset:2px}
input[aria-invalid=true],select[aria-invalid=true]{border-color:var(--danger)}
.err{color:var(--danger);font-size:13px;margin:0}

.seg{display:flex;background:var(--soft);border-radius:10px;padding:4px;gap:4px}
.seg input{position:absolute;opacity:0;pointer-events:none}
.seg label{flex:1;text-align:center;padding:9px 10px;border-radius:7px;font-size:14px;
  font-weight:500;color:var(--muted);cursor:pointer;transition:background .15s,color .15s}
.seg input:checked+label{background:var(--accent);color:#fff;font-weight:600;
  box-shadow:0 3px 8px rgba(79,70,229,.35)}
.seg input:focus-visible+label{outline:3px solid rgba(79,70,229,.35);outline-offset:1px}

.range-row{display:flex;align-items:center;gap:14px}
input[type=range]{flex:1;accent-color:var(--accent);height:28px;margin:0}
.range-val{min-width:84px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
.range-val small{font-weight:400;color:var(--muted)}

.actions{display:flex;gap:12px;flex-wrap:wrap;padding-top:8px}
button{font:inherit;cursor:pointer;border-radius:10px;height:48px;padding:0 24px;font-weight:600;
  border:1px solid transparent;transition:transform .1s,background .15s}
.primary{background:linear-gradient(135deg,#4F46E5,#7048E8);color:var(--accent-ink);box-shadow:0 8px 18px rgba(79,70,229,.30)}
.primary:hover{background:linear-gradient(135deg,#4338CA,#5F3DD6)}
.primary:active{transform:translateY(1px)}
.primary[disabled]{opacity:.65;cursor:progress}
.ghost{background:transparent;color:var(--ink);border-color:var(--line)}
.ghost:hover{background:var(--soft);border-color:#B8C0E0}

aside.panel{position:sticky;top:24px;padding:28px;min-height:320px;border-top:5px solid var(--line)}
aside.has-result{border-top-color:var(--c);background:linear-gradient(180deg,color-mix(in srgb,var(--c) 11%,#fff) 0%,#fff 62%)}
.empty h2,.result h2{font-family:var(--display);font-weight:500;font-size:21px;margin:0 0 8px}
.empty p{color:var(--muted);margin:0}
.empty .ghosts{margin-top:28px;display:grid;gap:14px}
.empty .ghosts span{display:block;height:12px;border-radius:6px;background:#E3E7F8}
.empty .ghosts span:nth-child(1){width:92%}.empty .ghosts span:nth-child(2){width:64%}
.empty .ghosts span:nth-child(3){width:38%}

.result .kicker{color:var(--muted);font-size:14px;margin:0 0 4px}
.result .verdict{font-family:var(--display);font-weight:700;font-size:44px;line-height:1.02;
  letter-spacing:-.025em;margin:0 0 6px}
.result .note{color:var(--ink);margin:0 0 22px;opacity:.85}
.meter{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:0 0 26px}
.meter div{height:10px;border-radius:999px;opacity:.22;transition:opacity .4s}
.meter div.on{opacity:1}
.meter-labels{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:-20px 0 26px;font-size:13px;color:var(--muted)}
.meter-labels span:nth-child(2){text-align:center}.meter-labels span:nth-child(3){text-align:right}
.bars{display:grid;gap:16px}
.bar-top{display:flex;justify-content:space-between;font-size:14px;margin-bottom:6px}
.bar-top b{font-weight:600}
.bar-top span{font-variant-numeric:tabular-nums;color:var(--muted)}
.track{height:10px;background:rgba(79,70,229,.10);border-radius:999px;overflow:hidden}
.fill{height:100%;width:0;border-radius:999px;transition:width .7s cubic-bezier(.2,.8,.2,1)}
.bar.top .bar-top b,.bar.top .bar-top span{color:var(--ink)}
.warns{margin:22px 0 0;padding:14px 16px;background:var(--warn-bg);color:var(--warn-ink);
  border-radius:10px;font-size:14px}
.warns p{margin:0 0 8px}.warns p:last-child{margin:0}
.disclaimer{margin:22px 0 0;font-size:13px;color:var(--muted)}
.banner{margin:0 0 4px;padding:12px 14px;border-radius:10px;background:#FDECEA;
  color:var(--danger);font-size:14px}
.hidden{display:none}

@media (max-width:960px){
  .layout{grid-template-columns:1fr}
  aside.panel{position:static;min-height:0}
}
@media (max-width:520px){
  .hero-in{padding:36px 16px 96px}
  .wrap{padding:0 16px 48px}
  form.panel{padding:4px 18px 22px}
  .actions button{flex:1}
}
@media (prefers-reduced-motion:reduce){
  *{transition:none!important}
}
</style>
</head>
<body>
<div class="hero">
  <div class="hero-in">
    <header>
      <h1 id="title"></h1>
      <p id="subtitle"></p>
    </header>
  </div>
</div>

<div class="wrap">
  <div class="layout">
    <form id="form" class="panel" novalidate></form>

    <aside id="side" class="panel" aria-live="polite">
      <div id="empty" class="empty">
        <h2>Your result will show here</h2>
        <p>Fill in the details and run the prediction.</p>
        <div class="ghosts" aria-hidden="true"><span></span><span></span><span></span></div>
      </div>
      <div id="result" class="result hidden"></div>
    </aside>
  </div>
</div>

<script>
const CFG = __CONFIG__;
document.getElementById('title').textContent = CFG.title;
document.getElementById('subtitle').textContent = CFG.subtitle;

const form = document.getElementById('form');
const emptyEl = document.getElementById('empty');
const resultEl = document.getElementById('result');

function el(tag, cls, text){
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
const fmt = v => String(v);

function buildField(f){
  const wrap = el('div', 'field');
  const id = 'f_' + f.name;
  const label = el('label', null, f.label);
  label.htmlFor = id;

  if (f.type === 'number'){
    const w = el('div', 'input-wrap');
    const i = document.createElement('input');
    i.type = 'number'; i.id = id; i.name = f.name; i.inputMode = 'decimal';
    i.min = f.min; i.max = f.max; i.step = f.step; i.value = f.default; i.required = true;
    w.append(i, el('span', 'unit', f.unit));
    wrap.append(label, w);
  } else if (f.type === 'select'){
    const s = document.createElement('select');
    s.id = id; s.name = f.name;
    f.options.forEach(([v, t]) => {
      const o = document.createElement('option');
      o.value = v; o.textContent = t; if (v === f.default) o.selected = true;
      s.append(o);
    });
    wrap.append(label, s);
  } else if (f.type === 'seg'){
    label.removeAttribute('for');
    label.className = 'lbl';
    const g = el('div', 'seg');
    g.setAttribute('role', 'radiogroup');
    g.setAttribute('aria-label', f.label);
    f.options.forEach(([v, t]) => {
      const rid = id + '_' + v;
      const r = document.createElement('input');
      r.type = 'radio'; r.name = f.name; r.id = rid; r.value = v;
      if (v === f.default) r.checked = true;
      const l = el('label', null, t); l.htmlFor = rid;
      g.append(r, l);
    });
    wrap.append(label, g);
  } else if (f.type === 'range'){
    const row = el('div', 'range-row');
    const i = document.createElement('input');
    i.type = 'range'; i.id = id; i.name = f.name;
    i.min = f.min; i.max = f.max; i.step = f.step; i.value = f.default;
    const out = el('div', 'range-val');
    const sync = () => { out.innerHTML = ''; out.append(fmt(i.value) + ' ', el('small', null, f.unit)); };
    i.addEventListener('input', sync); sync();
    row.append(i, out);
    wrap.append(label, row);
  }
  wrap.append(Object.assign(el('p', 'err hidden'), {id: 'err_' + f.name}));
  return wrap;
}

function buildForm(){
  const banner = el('p', 'banner hidden'); banner.id = 'banner';
  form.append(banner);
  const GROUP_COLORS = ['#4F46E5', '#0D9488', '#F97316'];
  [...new Set(CFG.fields.map(f => f.group))].forEach((g, gi) => {
    const sec = el('section', 'group');
    sec.style.setProperty('--g', GROUP_COLORS[gi % GROUP_COLORS.length]);
    sec.append(el('h2', null, g));
    const grid = el('div', 'grid');
    CFG.fields.filter(f => f.group === g).forEach(f => grid.append(buildField(f)));
    sec.append(grid);
    form.append(sec);
  });
  const actions = el('div', 'actions');
  const go = el('button', 'primary', 'Run prediction');
  go.type = 'submit'; go.id = 'go';
  const reset = el('button', 'ghost', 'Reset values');
  reset.type = 'button';
  reset.addEventListener('click', () => { form.reset(); form.querySelectorAll('input[type=range]').forEach(r => r.dispatchEvent(new Event('input'))); clearErrors(); showEmpty(); });
  actions.append(go, reset);
  form.append(actions);
}

function readValues(){
  const out = {};
  CFG.fields.forEach(f => {
    if (f.type === 'seg'){
      const c = form.querySelector('input[name="' + f.name + '"]:checked');
      out[f.name] = c ? c.value : '';
    } else {
      out[f.name] = form.elements[f.name].value;
    }
  });
  return out;
}

function clearErrors(){
  form.querySelectorAll('.err').forEach(e => { e.textContent = ''; e.classList.add('hidden'); });
  form.querySelectorAll('[aria-invalid]').forEach(e => e.removeAttribute('aria-invalid'));
  const b = document.getElementById('banner'); b.classList.add('hidden'); b.textContent = '';
}

function showErrors(errors){
  const names = Object.keys(errors);
  names.forEach(n => {
    const e = document.getElementById('err_' + n);
    if (e){ e.textContent = errors[n]; e.classList.remove('hidden'); }
    const input = form.elements[n];
    if (input && input.setAttribute) input.setAttribute('aria-invalid', 'true');
  });
  const b = document.getElementById('banner');
  b.textContent = 'Fix ' + names.length + (names.length === 1 ? ' field' : ' fields') + ' below and run the prediction again.';
  b.classList.remove('hidden');
  const first = document.getElementById('err_' + names[0]);
  if (first) first.scrollIntoView({behavior: 'smooth', block: 'center'});
}

const sideEl = document.getElementById('side');
function showEmpty(){
  sideEl.classList.remove('has-result');
  resultEl.classList.add('hidden'); emptyEl.classList.remove('hidden');
}

function showResult(data){
  const cls = CFG.classes[String(data.predicted)] || {name: 'Class ' + data.predicted, note: '', color: '#4F46E5', rank: 0};
  sideEl.style.setProperty('--c', cls.color);
  sideEl.classList.add('has-result');
  resultEl.innerHTML = '';
  resultEl.append(el('p', 'kicker', 'Your risk level'));
  const v = el('p', 'verdict', cls.name); v.style.color = cls.color;
  resultEl.append(v, el('p', 'note', cls.note));

  const ordered = Object.entries(CFG.classes).map(([k, c]) => ({k: Number(k), c})).sort((a, b) => a.c.rank - b.c.rank);
  const meter = el('div', 'meter');
  meter.setAttribute('role', 'img');
  meter.setAttribute('aria-label', 'Risk meter: ' + cls.name);
  ordered.forEach(o => {
    const seg = el('div', o.k === data.predicted ? 'on' : '');
    seg.style.background = o.c.color; meter.append(seg);
  });
  const ml = el('div', 'meter-labels');
  ordered.forEach(o => ml.append(el('span', null, o.c.name.replace(' risk', ''))));
  resultEl.append(meter, ml);

  const bars = el('div', 'bars');
  const best = Math.max(...data.scores.map(s => s.share));
  const rankOf = s => (CFG.classes[String(s['class'])] || {rank: 0}).rank;
  [...data.scores].sort((a, b) => rankOf(a) - rankOf(b)).forEach(s => {
    const c = CFG.classes[String(s['class'])] || {name: 'Class ' + s['class'], color: '#4F46E5'};
    const bar = el('div', 'bar' + (s.share === best ? ' top' : ''));
    const top = el('div', 'bar-top');
    top.append(el('b', null, c.name), el('span', null, s.share.toFixed(1) + '%'));
    const track = el('div', 'track'); const fill = el('div', 'fill');
    fill.style.background = c.color;
    track.append(fill); bar.append(top, track); bars.append(bar);
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = s.share + '%'; }));
  });
  resultEl.append(bars);

  if (data.warnings.length){
    const w = el('div', 'warns');
    data.warnings.forEach(t => w.append(el('p', null, t)));
    resultEl.append(w);
  }
  resultEl.append(el('p', 'disclaimer',
    'Confidence shows how the model\u2019s scores compare across the three risk levels. It is not a clinical probability. This tool supports screening and does not replace advice from a doctor.'));
  emptyEl.classList.add('hidden'); resultEl.classList.remove('hidden');
  if (window.matchMedia('(max-width:960px)').matches){
    resultEl.scrollIntoView({behavior: 'smooth', block: 'start'});
  }
}

form.addEventListener('submit', async ev => {
  ev.preventDefault();
  clearErrors();
  const btn = document.getElementById('go');
  btn.disabled = true; btn.textContent = 'Predicting\u2026';
  try {
    const res = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(readValues())
    });
    const data = await res.json();
    if (!res.ok || !data.ok){
      if (data.errors) showErrors(data.errors);
      else { const b = document.getElementById('banner'); b.textContent = 'Something went wrong on the server. Try again in a moment.'; b.classList.remove('hidden'); }
    } else showResult(data);
  } catch (e){
    const b = document.getElementById('banner');
    b.textContent = 'Could not reach the server. Check your connection and try again.';
    b.classList.remove('hidden');
  } finally {
    btn.disabled = false; btn.textContent = 'Run prediction';
  }
});

buildForm();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
