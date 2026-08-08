from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from shardsim.pipeline import ReferenceSample


def export_campaign_results(
    *,
    root: Path,
    reports_path: Path,
    spec: Mapping[str, Any],
    lock_payload: Mapping[str, Any],
    definitions: Sequence[Any],
    samples: Sequence[ReferenceSample],
    dataset_manifest_path: Path,
) -> dict[str, Any]:
    reports_path.mkdir(parents=True, exist_ok=True)
    definitions_by_id = {definition.case_id: definition for definition in definitions}
    samples_by_id = {sample.case_id: sample for sample in samples}
    ordered_samples = [
        samples_by_id[definition.case_id]
        for definition in definitions
        if definition.case_id in samples_by_id
    ]
    manifest_entries: dict[str, Mapping[str, Any]] = {}
    if dataset_manifest_path.is_file():
        manifest = _read_json(dataset_manifest_path)
        manifest_entries = {entry["case_id"]: entry for entry in manifest["samples"]}

    csv_path = reports_path / "results.csv"
    csv_buffer = io.StringIO(newline="")
    fieldnames = (
        "case_id",
        "split",
        "family",
        "alpha",
        "t_end",
        "center_x",
        "center_y",
        "sigma_x",
        "sigma_y",
        "amplitude",
        "baseline",
        "coarse_runtime_seconds",
        "nominal_runtime_seconds",
        "coarse_relative_l2",
        "coarse_mae",
        "coarse_rmse",
        "coarse_max_abs",
        "sample_sha256",
    )
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for sample in ordered_samples:
        definition = definitions_by_id[sample.case_id]
        writer.writerow(
            {
                "case_id": sample.case_id,
                "split": definition.split,
                "family": definition.family,
                "alpha": definition.alpha,
                "t_end": definition.t_end,
                "center_x": definition.center[0],
                "center_y": definition.center[1],
                "sigma_x": definition.sigma[0],
                "sigma_y": definition.sigma[1],
                "amplitude": definition.amplitude,
                "baseline": definition.baseline,
                "coarse_runtime_seconds": sample.coarse.runtime_seconds,
                "nominal_runtime_seconds": sample.nominal.runtime_seconds,
                "coarse_relative_l2": sample.metrics["relative_l2"],
                "coarse_mae": sample.metrics["mae"],
                "coarse_rmse": sample.metrics["rmse"],
                "coarse_max_abs": sample.metrics["max_abs_error"],
                "sample_sha256": manifest_entries.get(sample.case_id, {}).get("sha256"),
            }
        )
    _atomic_write_text(csv_path, csv_buffer.getvalue())

    coarse_shape = tuple(int(value) for value in spec["fidelity"]["coarse_shape"])
    nominal_shape = tuple(int(value) for value in spec["fidelity"]["nominal_shape"])
    metadata = {
        "export_version": 1,
        "campaign_name": spec["campaign_name"],
        "domain": spec["domain"],
        "equation": spec["equation"],
        "spec_sha256": lock_payload["spec_sha256"],
        "cases_sha256": lock_payload["cases_sha256"],
        "sample_count": len(ordered_samples),
        "case_order": [sample.case_id for sample in ordered_samples],
        "dtypes": {
            "identifiers": "unicode",
            "scalars_and_fields": "float64",
        },
        "units": {
            "alpha": "m^2/s",
            "t_end": "s",
            "extent": "m",
            "center": "relative [0,1]",
            "sigma": "relative [0,1]",
            "temperature_fields": "K",
        },
        "parameter_contract": {
            "centers": ["sample", "x", "y"],
            "sigmas": ["sample", "x", "y"],
            "extents": ["sample", "x", "y"],
            "boundaries": ["sample", "top", "bottom", "left", "right"],
        },
        "array_contract": {
            "coarse_fields": ["sample", "y_coarse", "x_coarse"],
            "nominal_fields": ["sample", "y_nominal", "x_nominal"],
            "coarse_on_nominal": ["sample", "y_nominal", "x_nominal"],
            "deltas": ["sample", "y_nominal", "x_nominal"],
            "error_maps": ["sample", "y_nominal", "x_nominal"],
        },
    }
    npz_path = reports_path / "combined-results.npz"
    temporary_npz = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_npz,
        case_ids=np.asarray([sample.case_id for sample in ordered_samples]),
        splits=np.asarray(
            [definitions_by_id[sample.case_id].split for sample in ordered_samples]
        ),
        families=np.asarray(
            [definitions_by_id[sample.case_id].family for sample in ordered_samples]
        ),
        alphas=np.asarray(
            [definitions_by_id[sample.case_id].alpha for sample in ordered_samples],
            dtype=np.float64,
        ),
        t_ends=np.asarray(
            [definitions_by_id[sample.case_id].t_end for sample in ordered_samples],
            dtype=np.float64,
        ),
        centers=_parameter_matrix(
            [definitions_by_id[sample.case_id].center for sample in ordered_samples], 2
        ),
        sigmas=_parameter_matrix(
            [definitions_by_id[sample.case_id].sigma for sample in ordered_samples], 2
        ),
        extents=_parameter_matrix(
            [definitions_by_id[sample.case_id].extent for sample in ordered_samples], 2
        ),
        amplitudes=np.asarray(
            [definitions_by_id[sample.case_id].amplitude for sample in ordered_samples],
            dtype=np.float64,
        ),
        baselines=np.asarray(
            [definitions_by_id[sample.case_id].baseline for sample in ordered_samples],
            dtype=np.float64,
        ),
        boundaries=_parameter_matrix(
            [
                (
                    definitions_by_id[sample.case_id].boundaries.top,
                    definitions_by_id[sample.case_id].boundaries.bottom,
                    definitions_by_id[sample.case_id].boundaries.left,
                    definitions_by_id[sample.case_id].boundaries.right,
                )
                for sample in ordered_samples
            ],
            4,
        ),
        coarse_fields=_stack_or_empty(
            [sample.coarse.field for sample in ordered_samples], coarse_shape
        ),
        nominal_fields=_stack_or_empty(
            [sample.nominal.field for sample in ordered_samples], nominal_shape
        ),
        coarse_on_nominal=_stack_or_empty(
            [sample.coarse_on_nominal for sample in ordered_samples], nominal_shape
        ),
        deltas=_stack_or_empty(
            [sample.delta for sample in ordered_samples], nominal_shape
        ),
        error_maps=_stack_or_empty(
            [sample.error_map for sample in ordered_samples], nominal_shape
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary_npz.replace(npz_path)
    export_manifest = {
        **metadata,
        "csv_path": csv_path.relative_to(root).as_posix(),
        "csv_sha256": _sha256_file(csv_path),
        "npz_path": npz_path.relative_to(root).as_posix(),
        "npz_sha256": _sha256_file(npz_path),
        "npz_content_sha256": _npz_content_sha256(npz_path),
    }
    manifest_path = reports_path / "export.manifest.json"
    _atomic_write_text(
        manifest_path,
        json.dumps(export_manifest, indent=2, sort_keys=True) + "\n",
    )
    export_manifest["manifest_path"] = manifest_path.relative_to(root).as_posix()
    return export_manifest


def render_campaign_dashboard(
    *,
    reports_path: Path,
    campaign_status: Mapping[str, Any],
    samples: Sequence[ReferenceSample],
    definitions: Sequence[Any],
    models: Sequence[Mapping[str, Any]],
    run_records: Sequence[Mapping[str, Any]],
    export: Mapping[str, Any],
) -> Path:
    definitions_by_id = {definition.case_id: definition for definition in definitions}
    dashboard_models = []
    for model in models:
        key = str(model["reproducibility_key"])
        evaluations = {}
        evaluation_path = reports_path / "models" / key
        if evaluation_path.exists():
            for path in sorted(evaluation_path.glob("evaluation-*.json")):
                report = _read_json(path)
                evaluations[str(report["split"])] = {
                    "passed": report["passed"],
                    "metrics": report["metrics"],
                    "case_count": len(report["cases"]),
                    "checkpoint_comparison": report.get("checkpoint_comparison"),
                }
        dashboard_models.append(
            {
                "key": key,
                "lineage_id": model.get("lineage_id"),
                "lineage_name": model.get("lineage_name", model.get("model_id")),
                "checkpoint_index": int(model.get("checkpoint_index", 0)),
                "parent_key": model.get("parent_reproducibility_key"),
                "training_mode": model.get("training_mode", "legacy"),
                "algorithm": model["algorithm"],
                "implementation_algorithm": model.get("implementation_algorithm"),
                "active": bool(model.get("active")),
                "trained_at": model.get("trained_at_utc"),
                "training_case_count": len(model.get("training_case_ids", ())),
                "new_training_case_count": len(
                    model.get("new_training_case_ids", model.get("training_case_ids", ()))
                ),
                "artifact_content_sha256": model.get("artifact_content_sha256"),
                "evaluations": evaluations,
            }
        )
    dashboard_models.sort(
        key=lambda model: (model["lineage_name"] or "", model["checkpoint_index"])
    )
    cases = []
    for sample in samples:
        definition = definitions_by_id[sample.case_id]
        cases.append(
            {
                "case_id": sample.case_id,
                "split": definition.split,
                "family": definition.family,
                "parameters": {
                    "alpha": definition.alpha,
                    "t_end": definition.t_end,
                    "center": list(definition.center),
                    "sigma": list(definition.sigma),
                    "amplitude": definition.amplitude,
                    "baseline": definition.baseline,
                },
                "metrics": dict(sample.metrics),
                "runtime": {
                    "coarse": sample.coarse.runtime_seconds,
                    "nominal": sample.nominal.runtime_seconds,
                },
                "coarse_on_nominal": _rounded_list(sample.coarse_on_nominal),
                "nominal": _rounded_list(sample.nominal.field),
                "delta": _rounded_list(sample.delta),
                "error_map": _rounded_list(sample.error_map),
            }
        )
    runs = [
        {
            "run_id": record.get("run_id"),
            "status": record.get("status"),
            "started_at": record.get("started_at_utc"),
            "completed": len(record.get("completed_case_ids", ())),
            "failures": len(record.get("failures", ())),
            "selection": record.get("selection", {}),
        }
        for record in reversed(run_records)
    ]
    payload = {
        "status": campaign_status,
        "cases": cases,
        "models": dashboard_models,
        "runs": runs,
        "export": {
            "sample_count": export["sample_count"],
            "csv_sha256": export["csv_sha256"],
            "npz_content_sha256": export["npz_content_sha256"],
        },
    }
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    dashboard_path = reports_path / "dashboard.html"
    _atomic_write_text(dashboard_path, _dashboard_html(serialized))
    return dashboard_path


def _dashboard_html(serialized_payload: str) -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ShardSim — résultats de campagne</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f4f7fb; --panel:#fff; --text:#172033; --muted:#64748b; --line:#dbe3ef; --accent:#0969da; --good:#16803c; --bad:#b42318; --shadow:0 10px 30px rgba(23,32,51,.08); }}
    @media (prefers-color-scheme:dark) {{ :root {{ --bg:#0f1724; --panel:#172033; --text:#edf3fc; --muted:#a7b4c7; --line:#344055; --accent:#63a7ff; --good:#5bd17d; --bad:#ff8b82; --shadow:none; }} }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--text); }}
    header {{ padding:28px clamp(18px,4vw,56px) 18px; background:linear-gradient(120deg,#0b3b73,#0969da); color:white; }}
    h1 {{ margin:0 0 4px; font-size:clamp(25px,4vw,40px); }} h2 {{ margin:0 0 16px; font-size:20px; }} p {{ margin:4px 0; }}
    main {{ max-width:1500px; margin:auto; padding:22px clamp(14px,3vw,42px) 48px; }}
    .grid {{ display:grid; grid-template-columns:repeat(12,1fr); gap:16px; }} .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; box-shadow:var(--shadow); }}
    .summary {{ grid-column:span 3; }} .controls,.fields,.learning,.models,.runs {{ grid-column:1/-1; }} .details {{ grid-column:span 4; }} .plot {{ grid-column:span 4; }}
    .value {{ font-size:30px; font-weight:750; }} .muted {{ color:var(--muted); }} .good {{ color:var(--good); }} .bad {{ color:var(--bad); }}
    .toolbar {{ display:flex; gap:12px; flex-wrap:wrap; align-items:end; }} label {{ display:grid; gap:5px; font-weight:650; }} select {{ min-width:190px; padding:9px 12px; border:1px solid var(--line); border-radius:8px; background:var(--panel); color:var(--text); }}
    a.button {{ display:inline-block; padding:10px 14px; color:white; background:var(--accent); border-radius:8px; text-decoration:none; font-weight:700; }}
    canvas {{ width:100%; aspect-ratio:1; image-rendering:pixelated; border-radius:8px; background:#111827; }}
    canvas.chart {{ aspect-ratio:3.5/1; min-height:260px; image-rendering:auto; background:transparent; }}
    dl {{ display:grid; grid-template-columns:minmax(120px,1fr) 2fr; gap:8px 12px; margin:0; }} dt {{ color:var(--muted); }} dd {{ margin:0; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
    .table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; white-space:nowrap; }} th,td {{ padding:10px 12px; text-align:left; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }} code {{ font-size:12px; }}
    .badge {{ padding:3px 8px; border-radius:999px; background:color-mix(in srgb,var(--accent) 14%,transparent); color:var(--accent); font-weight:700; }}
    .empty {{ padding:24px; text-align:center; color:var(--muted); }}
    @media(max-width:900px) {{ .summary,.details,.plot {{ grid-column:1/-1; }} }}
  </style>
</head>
<body>
<header><h1 id="campaign-name">ShardSim</h1><p>Résultats concaténés, comparaisons et modèles reproductibles</p></header>
<main class="grid">
  <section class="panel summary"><div class="muted">Cas terminés</div><div class="value" id="completed">0</div></section>
  <section class="panel summary"><div class="muted">Cas restants</div><div class="value" id="pending">0</div></section>
  <section class="panel summary"><div class="muted">Modèles enregistrés</div><div class="value" id="model-count">0</div></section>
  <section class="panel summary"><div class="muted">Modèle actif</div><div class="value" id="active-model" style="font-size:18px">Aucun</div></section>
  <section class="panel controls">
    <h2>Explorer les simulations</h2>
    <div class="toolbar">
      <label>Split<select id="split-filter"><option value="">Tous</option></select></label>
      <label>Famille<select id="family-filter"><option value="">Toutes</option></select></label>
      <label>Cas<select id="case-select"></select></label>
      <a class="button" href="results.csv">Ouvrir le tableau CSV</a>
      <a class="button" href="combined-results.npz">Réutiliser le dataset ML</a>
    </div>
    <p class="muted" id="filter-message"></p>
  </section>
  <section class="panel details"><h2>Cas sélectionné</h2><dl id="case-details"></dl></section>
  <section class="panel plot"><h2>Pré-simulation grossière</h2><canvas id="coarse"></canvas><p class="muted" id="coarse-range"></p></section>
  <section class="panel plot"><h2>Calcul nominal</h2><canvas id="nominal"></canvas><p class="muted" id="nominal-range"></p></section>
  <section class="panel plot"><h2>Correction nominale</h2><canvas id="delta"></canvas><p class="muted" id="delta-range"></p></section>
  <section class="panel plot"><h2>Erreur absolue</h2><canvas id="error"></canvas><p class="muted" id="error-range"></p></section>
  <section class="panel learning"><h2>Courbe d’apprentissage cumulative</h2><p class="muted">Chaque point représente un checkpoint de la même lignée. Plus l’erreur descend lorsque le nombre de cas augmente, plus l’apprentissage progresse.</p><canvas class="chart" id="learning-chart"></canvas><p class="muted" id="learning-message"></p></section>
  <section class="panel models"><h2>Comparer et réutiliser les checkpoints</h2><p class="muted">Les checkpoints restent immuables pour l’audit, mais appartiennent à une même lignée cumulative. Le modèle actif est celui utilisé par les prochaines évaluations et prévisualisations.</p><div class="table-wrap"><table><thead><tr><th>État</th><th>Algorithme</th><th>Checkpoint</th><th>Clé</th><th>Cas cumulés</th><th>Nouveaux</th><th>Validation L2</th><th>Gain/grossier</th><th>Gradient L2</th><th>Couverture 2σ</th><th>Accélération</th></tr></thead><tbody id="models-body"></tbody></table></div></section>
  <section class="panel runs"><h2>Historique des lancements</h2><div class="table-wrap"><table><thead><tr><th>Lancement</th><th>État</th><th>Début</th><th>Cas terminés</th><th>Erreurs</th><th>Sélection</th></tr></thead><tbody id="runs-body"></tbody></table></div></section>
</main>
<script>
const data={serialized_payload};
const $=id=>document.getElementById(id); const fmt=n=>Number.isFinite(Number(n))?Number(n).toLocaleString('fr-FR',{{maximumSignificantDigits:6}}):'—';
const textCell=(row,value)=>{{const cell=row.insertCell();cell.textContent=value;return cell;}};
$('campaign-name').textContent=data.status.campaign_name; $('completed').textContent=data.status.completed_cases; $('pending').textContent=data.status.pending_cases; $('model-count').textContent=data.models.length;
const active=data.models.find(model=>model.active); $('active-model').textContent=active?active.algorithm+' · '+active.key.slice(0,10):'Aucun';
for(const split of [...new Set(data.cases.map(item=>item.split))].sort()) $('split-filter').add(new Option(split,split));
for(const family of [...new Set(data.cases.map(item=>item.family))].sort()) $('family-filter').add(new Option(family,family));
function filteredCases(){{return data.cases.filter(item=>(!$('split-filter').value||item.split===$('split-filter').value)&&(!$('family-filter').value||item.family===$('family-filter').value));}}
function refreshCases(){{const previous=$('case-select').value; $('case-select').replaceChildren(); const cases=filteredCases(); for(const item of cases) $('case-select').add(new Option(item.case_id,item.case_id)); if(cases.some(item=>item.case_id===previous)) $('case-select').value=previous; $('filter-message').textContent=cases.length+' cas disponible(s)'; renderCase();}}
function heatColor(t){{t=Math.max(0,Math.min(1,t)); const stops=[[13,8,135],[126,3,168],[204,71,120],[248,149,64],[240,249,33]]; const p=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(p)),f=p-i; return stops[i].map((v,j)=>Math.round(v+(stops[i+1][j]-v)*f));}}
function draw(canvasId,rangeId,field,symmetric=false){{const canvas=$(canvasId),ctx=canvas.getContext('2d'); if(!field){{canvas.width=1;canvas.height=1;ctx.clearRect(0,0,1,1);$(rangeId).textContent='Aucune donnée';return;}} const h=field.length,w=field[0].length,flat=field.flat(),rawMin=Math.min(...flat),rawMax=Math.max(...flat),limit=Math.max(Math.abs(rawMin),Math.abs(rawMax)),min=symmetric?-limit:rawMin,max=symmetric?limit:rawMax; canvas.width=w;canvas.height=h; const image=ctx.createImageData(w,h); for(let y=0;y<h;y++)for(let x=0;x<w;x++){{const t=(field[h-1-y][x]-min)/Math.max(max-min,1e-15),color=heatColor(t),offset=(y*w+x)*4; image.data[offset]=color[0];image.data[offset+1]=color[1];image.data[offset+2]=color[2];image.data[offset+3]=255;}} ctx.putImageData(image,0,0); $(rangeId).textContent='min '+fmt(rawMin)+' · max '+fmt(rawMax);}}
function renderCase(){{const item=data.cases.find(entry=>entry.case_id===$('case-select').value); const details=$('case-details');details.replaceChildren(); if(!item){{details.innerHTML='<div class="empty">Lancez au moins un cas pour afficher les cartes.</div>'; draw('coarse','coarse-range');draw('nominal','nominal-range');draw('delta','delta-range');draw('error','error-range');return;}} const rows=[['Identifiant',item.case_id],['Split / famille',item.split+' / '+item.family],['Diffusivité α',fmt(item.parameters.alpha)],['Temps final',fmt(item.parameters.t_end)],['Centre',item.parameters.center.map(fmt).join(', ')],['Sigma',item.parameters.sigma.map(fmt).join(', ')],['Erreur relative L2',fmt(item.metrics.relative_l2)],['MAE grossier',fmt(item.metrics.mae)],['Temps grossier',fmt(item.runtime.coarse)+' s'],['Temps nominal',fmt(item.runtime.nominal)+' s']]; for(const [key,value] of rows){{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;details.append(dt,dd);}} draw('coarse','coarse-range',item.coarse_on_nominal);draw('nominal','nominal-range',item.nominal);draw('delta','delta-range',item.delta,true);draw('error','error-range',item.error_map);}}
for(const id of ['split-filter','family-filter']) $(id).addEventListener('change',refreshCases); $('case-select').addEventListener('change',renderCase); refreshCases();
function drawLearningChart(){{
  const canvas=$('learning-chart'),ctx=canvas.getContext('2d');
  const points=data.models.filter(model=>Number.isFinite(model.evaluations.validation?.metrics?.mean_preview_relative_l2));
  canvas.width=1100;canvas.height=330;ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!points.length){{$('learning-message').textContent='Évaluez au moins un checkpoint sur le split validation pour afficher la progression.';return;}}
  const groups=new Map();
  for(const point of points){{const name=point.lineage_name||point.algorithm;if(!groups.has(name))groups.set(name,[]);groups.get(name).push(point);}}
  for(const values of groups.values())values.sort((left,right)=>left.checkpoint_index-right.checkpoint_index);
  const margin={{left:75,right:30,top:45,bottom:55}},width=canvas.width-margin.left-margin.right,height=canvas.height-margin.top-margin.bottom;
  const xValues=points.map(point=>point.training_case_count),allErrors=points.flatMap(point=>[point.evaluations.validation.metrics.mean_preview_relative_l2,point.evaluations.validation.metrics.mean_coarse_relative_l2]);
  const xMin=Math.min(...xValues),xMax=Math.max(...xValues),yMax=Math.max(...allErrors)*1.12||1;
  const x=value=>margin.left+(value-xMin)/Math.max(xMax-xMin,1)*width,y=value=>margin.top+(1-value/yMax)*height;
  ctx.font='13px system-ui';ctx.strokeStyle='#94a3b8';ctx.fillStyle='#64748b';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(margin.left,margin.top);ctx.lineTo(margin.left,margin.top+height);ctx.lineTo(margin.left+width,margin.top+height);ctx.stroke();
  for(let index=0;index<=4;index++){{const value=yMax*index/4,position=y(value);ctx.fillText(fmt(value),8,position+5);ctx.strokeStyle='rgba(148,163,184,.25)';ctx.beginPath();ctx.moveTo(margin.left,position);ctx.lineTo(margin.left+width,position);ctx.stroke();}}
  const colors=['#0969da','#d97706','#16803c','#9333ea','#dc2626'];let colorIndex=0;
  for(const [name,values] of groups){{const color=colors[colorIndex++%colors.length];ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=3;ctx.beginPath();values.forEach((point,index)=>{{const px=x(point.training_case_count),py=y(point.evaluations.validation.metrics.mean_preview_relative_l2);if(index===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);}});ctx.stroke();values.forEach(point=>{{const px=x(point.training_case_count),py=y(point.evaluations.validation.metrics.mean_preview_relative_l2);ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);ctx.fill();ctx.fillText(String(point.training_case_count),px-5,margin.top+height+24);}});ctx.fillText(name,margin.left+colorIndex*180-170,20);}}
  const coarse=[...points].sort((left,right)=>left.training_case_count-right.training_case_count);ctx.setLineDash([6,5]);ctx.strokeStyle='#94a3b8';ctx.lineWidth=2;ctx.beginPath();coarse.forEach((point,index)=>{{const px=x(point.training_case_count),py=y(point.evaluations.validation.metrics.mean_coarse_relative_l2);if(index===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);}});ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#64748b';ctx.fillText('Grossier',margin.left+groups.size*180+10,20);ctx.fillText('Nombre de cas cumulés',margin.left+width/2-70,canvas.height-8);
  $('learning-message').textContent=points.length+' checkpoint(s), '+groups.size+' lignée(s) évaluée(s).';
}}
drawLearningChart();
const modelsBody=$('models-body'); if(!data.models.length){{const row=modelsBody.insertRow();const cell=row.insertCell();cell.colSpan=11;cell.className='empty';cell.textContent='Aucun modèle entraîné.';}} for(const model of data.models){{const metrics=model.evaluations.validation?.metrics||{{}},row=modelsBody.insertRow();const state=textCell(row,model.active?'ACTIF':'Archivé');state.className=model.active?'good':'';textCell(row,model.algorithm);textCell(row,'#'+(model.checkpoint_index||'?')+' · '+model.training_mode);const key=textCell(row,model.key.slice(0,16));key.title=model.key;textCell(row,model.training_case_count);textCell(row,model.new_training_case_count);textCell(row,fmt(metrics.mean_preview_relative_l2));textCell(row,fmt(metrics.mean_relative_gain_vs_coarse));textCell(row,fmt(metrics.mean_gradient_relative_l2));textCell(row,fmt(metrics.mean_coverage_2sigma));textCell(row,fmt(metrics.median_preview_speedup)+'×');}}
const runsBody=$('runs-body'); if(!data.runs.length){{const row=runsBody.insertRow();const cell=row.insertCell();cell.colSpan=6;cell.className='empty';cell.textContent='Aucun lancement enregistré.';}} for(const run of data.runs){{const row=runsBody.insertRow();textCell(row,run.run_id||'—');textCell(row,run.status||'—');textCell(row,run.started_at||'—');textCell(row,run.completed);textCell(row,run.failures);const selection=Object.entries(run.selection||{{}}).filter(([,v])=>v&&(!Array.isArray(v)||v.length)).map(([k,v])=>k+'='+v).join(' · ');textCell(row,selection||'tous');}}
</script>
</body>
</html>
"""


def _stack_or_empty(arrays: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not arrays:
        return np.empty((0, *shape), dtype=np.float64)
    return np.stack(arrays).astype(np.float64, copy=False)


def _parameter_matrix(values: Sequence[Sequence[float]], width: int) -> np.ndarray:
    if not values:
        return np.empty((0, width), dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(len(values), width)


def _rounded_list(array: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(array, dtype=np.float64), decimals=7).tolist()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = np.asarray(archive[name])
            digest.update(name.encode("utf-8"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
