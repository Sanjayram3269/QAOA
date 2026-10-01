const DATA = {
  main: '../data/ml/paper_results/01_main_results.csv',
  budget: '../data/ml/paper_results/02_budget_results.csv',
  noise: '../data/ml/paper_results/03_noise_results.csv',
  budgetNoise: '../data/ml/paper_results/04_budget_noise_results.csv',
  validation: '../data/ml/paper_results/05_model_validation_comparison.csv',
  ablation: '../data/ml/paper_results/06_feature_ablation.csv',
  importance: '../data/ml/paper_results/07_permutation_importance.csv',
  stats: '../data/ml/paper_results/08_statistical_tests.csv',
  selected: '../data/ml/paper_results/09_test_selected_configurations.csv',
  fixed: '../data/ml/paper_results/10_fixed_baseline_choices.csv',
  features: '../data/ml/paper_results/11_test_graph_features.csv',
  metadata: '../data/ml/paper_results/12_experiment_metadata.csv',
  graphs: '../data/graphs/master_graph_manifest.csv',
  split: '../data/ml/splits/graph_split_assignments.csv',
  datasetManifest: '../data/ml/dataset_manifest.json',
  analysisManifest: '../data/ml/final_selector/analysis_manifest.json',
  experiment: '../config/experiment.yaml'
};

const CONFIGS = [
  ['C01',1,'COBYLA',256],['C02',1,'COBYLA',512],['C03',1,'SPSA',256],['C04',1,'SPSA',512],
  ['C05',2,'COBYLA',256],['C06',2,'COBYLA',512],['C07',2,'SPSA',256],['C08',2,'SPSA',512],
  ['C09',3,'COBYLA',256],['C10',3,'COBYLA',512],['C11',3,'SPSA',256],['C12',3,'SPSA',512]
].map(([config_id,depth,optimizer,shots])=>({config_id,depth,optimizer,shots}));

const state = {
  page:'overview', data:{}, loaded:false, budget:'ALL', noise:'ALL', graph:'ALL', search:'', selectedGraph:null,
  theme:'dark'
};

const $ = (s,root=document)=>root.querySelector(s);
const $$ = (s,root=document)=>[...root.querySelectorAll(s)];
const fmt = (n,d=4)=>Number.isFinite(Number(n)) ? Number(n).toFixed(d) : '—';
const pct = (n,d=2)=>Number.isFinite(Number(n)) ? `${(Number(n)*100).toFixed(d)}%` : '—';
const signed = (n,d=4)=>Number(n)>=0?`+${Number(n).toFixed(d)}`:Number(n).toFixed(d);
const esc = s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
const mean = a=>a.length?a.reduce((x,y)=>x+Number(y||0),0)/a.length:NaN;

function parseCSV(text){
  const rows=[]; let row=[], cell='', quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i], n=text[i+1];
    if(c==='"') { if(quoted && n==='"'){cell+='"';i++;} else quoted=!quoted; }
    else if(c===','&&!quoted){row.push(cell);cell='';}
    else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&n==='\n')i++;row.push(cell);if(row.some(v=>v!==''))rows.push(row);row=[];cell='';}
    else cell+=c;
  }
  if(cell!==''||row.length){row.push(cell);if(row.some(v=>v!==''))rows.push(row);}
  if(!rows.length)return [];
  const headers=rows.shift().map(h=>h.trim());
  return rows.map(r=>Object.fromEntries(headers.map((h,i)=>[h,(r[i]??'').trim()])));
}

async function load(path){const r=await fetch(path,{cache:'no-store'});if(!r.ok)throw new Error(`${r.status} ${path}`);return r.text();}
async function loadAll(){
  const entries=Object.entries(DATA); const out={};
  await Promise.all(entries.map(async([k,p])=>{try{const raw=await load(p);out[k]=p.endsWith('.json')?JSON.parse(raw):p.endsWith('.yaml')?raw:parseCSV(raw);}catch(e){out[k]=[];console.warn('Artifact unavailable',k,e.message)}}));
  state.data=out; state.loaded=true;
}

function chart(id,traces,layout={}){
  const el=document.getElementById(id); if(!el)return;
  if(!window.Plotly){el.innerHTML='<div class="empty">Chart engine unavailable. The data tables remain fully usable.</div>';return;}
  Plotly.newPlot(el,traces,{paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{family:'Manrope, sans-serif',color:'#8998ad',size:10},margin:{l:45,r:20,t:12,b:40},xaxis:{gridcolor:'#172131',zerolinecolor:'#263449'},yaxis:{gridcolor:'#172131',zerolinecolor:'#263449'},hoverlabel:{bgcolor:'#0c1420',bordercolor:'#2b3c55',font:{color:'#fff'}},showlegend:true,legend:{orientation:'h',y:1.12,x:0},...layout},{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']});
}
function pageTitle(eyebrow,title,desc,tools=''){return `<div class="page-title"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${desc}</p></div><div class="page-tools">${tools}</div></div>`}
function kpi(label,value,sub='',cls=''){return `<div class="kpi ${cls}"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-sub">${sub}</div></div>`}
function panel(title,sub,body,cls=''){return `<div class="panel ${cls}"><div class="panel-head"><div><div class="panel-title">${title}</div>${sub?`<div class="panel-sub">${sub}</div>`:''}</div></div>${body}</div>`}
function seg(name,items,current){return `<div class="seg" data-seg="${name}">${items.map(x=>`<button class="${x===current?'active':''}" data-value="${x}">${x}</button>`).join('')}</div>`}
function rowStatus(v){return `<span class="status ${v?'ok':'warn'}">${v?'MATCH':'GAP'}</span>`}

function currentMain(){return state.data.main?.[0]||{};}
function currentBudget(){return state.data.budget||[];}
function currentNoise(){return state.data.noise||[];}
function currentBN(){return state.data.budgetNoise||[];}
function filteredSelected(){return (state.data.selected||[]).filter(r=>(state.budget==='ALL'||r.budget===state.budget)&&(state.noise==='ALL'||r.noise_condition===state.noise)&&(state.graph==='ALL'||r.graph_id===state.graph));}
function graphMap(){const m={};(state.data.graphs||[]).forEach(r=>m[r.graph_id]=r);return m;}
function selectedGraphs(){return [...new Set((state.data.selected||[]).map(r=>r.graph_id))];}

function renderOverview(){
  const m=currentMain();
  $('#page-overview').innerHTML=pageTitle('QUANTUM INTELLIGENCE / OVERVIEW','Resource-Aware QAOA Configuration Intelligence','A research-grade view of the frozen MaxCut experiment: graph-aware selection, resource budgets, controlled noise, and ML evidence.',seg('overviewBudget',['ALL','B256','B512'],state.budget))+`
  <div class="hero"><div class="hero-orbit"></div><div class="hero-grid"><div><div class="eyebrow">FROZEN TEST EVALUATION</div><h2>Choose the QAOA configuration before you execute it.</h2><div class="hero-copy">The selector maps graph structure, configuration descriptors and execution conditions into a ranked configuration decision. This dashboard visualizes the frozen research artifacts without recomputing or altering the underlying experiment.</div><div class="hero-badges"><span class="badge cyan">MAXCUT</span><span class="badge">14 TEST GRAPHS</span><span class="badge">84 TEST CASES</span><span class="badge green">EXTRA TREES</span><span class="badge">SEED 2027</span></div></div><div class="hero-metric"><div class="metric-label">ML selected expected approximation ratio</div><div class="hero-number">${fmt(m.ml_mean,5)} <small>/ 1.00</small></div><div class="delta">${signed(m.ml_minus_fixed,6)} vs fixed · ${signed(m.relative_gain_percent,3)}%</div></div></div></div>
  <div class="grid kpi-grid">${kpi('ML selector',fmt(m.ml_mean,5),'84 frozen cases','accent')}${kpi('Fixed baseline',fmt(m.fixed_mean,5),'task-aligned reference')}${kpi('Random expectation',fmt(m.random_expected_mean,5),'configuration expectation')}${kpi('Oracle reference',fmt(m.oracle_mean,5),'best observed config')}${kpi('Mean regret',fmt(m.mean_regret,5),'distance to oracle')}${kpi('Oracle top-3',pct(m.oracle_top3_accuracy),'selection coverage','green')}</div>
  <div class="grid two"><div>${panel('Selector vs reference','Aggregate expected approximation ratio',`<div id="overviewBars" class="chart"></div>`)}</div><div>${panel('Research signal','The number matters less than the evidence around it',`<div class="metric-pair"><div class="metric-box"><div class="label">ML − fixed</div><div class="value ${Number(m.ml_minus_fixed)>=0?'positive':'negative'}">${signed(m.ml_minus_fixed,6)}</div></div><div class="metric-box"><div class="label">Graph fraction improved</div><div class="value">${pct(m.ml_better_graph_fraction||0)}</div></div></div><div class="section-gap callout"><strong>Interpretation.</strong> The aggregate ML/fixed difference is small. The frozen statistical analysis should be read alongside the confidence interval and sign-flip test rather than as a standalone claim of superiority.</div><div class="section-gap research-note">The dashboard is a presentation and inspection layer over committed artifacts; it does not silently rerun QAOA.</div>`)}</div></div>
  <div class="grid two section-gap"><div>${panel('Budget × noise landscape','Frozen expected approximation ratios across resource/noise cells',`<div id="overviewHeat" class="chart"></div>`)}</div><div>${panel('Experiment contract','What is actually frozen',`<div class="timeline"><div class="step"><b>Graph population</b><small>89 complete graphs admitted to selector analysis.</small></div><div class="step"><b>Split</b><small>62 train · 13 validation · 14 test.</small></div><div class="step"><b>Search space</b><small>12 configurations across depth, optimizer and shots.</small></div><div class="step"><b>Conditions</b><small>B256/B512 × N0/N1/N2.</small></div></div><div class="callout"><strong>Primary metric:</strong> expected approximation ratio. <strong>Primary task:</strong> MaxCut configuration selection.</div>`)}</div></div>`;
  renderOverviewCharts();
}
function renderOverviewCharts(){
  const m=currentMain(); chart('overviewBars',[{x:['ML selector','Fixed baseline','Random','Oracle'],y:[+m.ml_mean,+m.fixed_mean,+m.random_expected_mean,+m.oracle_mean],type:'bar',marker:{color:['#63e6ff','#5e6c80','#7e6ab9','#50e3a4']},hovertemplate:'%{x}: %{y:.5f}<extra></extra>'}],{yaxis:{title:'Expected approximation ratio',range:[0.65,.76],tickformat:'.2f'},showlegend:false});
  const cells=currentBN(); const xs=['B256_N0','B256_N1','B256_N2','B512_N0','B512_N1','B512_N2']; chart('overviewHeat',[{x:['N0','N1','N2'],y:['B256','B512'],z:[[...xs.slice(0,3).map(k=>+(cells.find(r=>r.scope===k)?.ml_mean||0))],[...xs.slice(3).map(k=>+(cells.find(r=>r.scope===k)?.ml_mean||0))]],type:'heatmap',colorscale:[[0,'#111b2a'],[.5,'#15566a'],[1,'#63e6ff']],hovertemplate:'%{y} / %{x}: %{z:.5f}<extra></extra>'}],{xaxis:{title:'Noise condition'},yaxis:{title:'Budget'}});
}

function renderPerformance(){
  const b=currentBudget(), n=currentNoise();
  $('#page-performance').innerHTML=pageTitle('ANALYSIS / PERFORMANCE','Performance Observatory','Compare ML-selected, fixed, random-expectation and oracle references across resource budgets and noise environments.',seg('perfBudget',['ALL','B256','B512'],state.budget))+`
  <div class="grid kpi-grid">${kpi('All-case ML',fmt(currentMain().ml_mean,5),'aggregate')}${kpi('All-case fixed',fmt(currentMain().fixed_mean,5),'reference')}${kpi('B512 ML',fmt(b.find(r=>r.scope==='B512')?.ml_mean,5),'42 cases','accent')}${kpi('B512 ML − fixed',signed(b.find(r=>r.scope==='B512')?.ml_minus_fixed,5),'resource response')}${kpi('N0 ML',fmt(n.find(r=>r.scope==='N0')?.ml_mean,5),'ideal simulation')}${kpi('N2 ML',fmt(n.find(r=>r.scope==='N2')?.ml_mean,5),'higher controlled noise')}</div>
  <div class="grid two"><div>${panel('Performance by budget','42 test cases per budget',`<div id="budgetChart" class="chart"></div>`)}</div><div>${panel('Performance by noise','28 test cases per condition',`<div id="noiseChart" class="chart"></div>`)}</div></div>
  <div class="grid two section-gap"><div>${panel('Budget evidence','Exact frozen values',table(b,['scope','graphs','cases','ml_mean','fixed_mean','random_expected_mean','oracle_mean','ml_minus_fixed','relative_gain_percent'],['Scope','Graphs','Cases','ML','Fixed','Random','Oracle','Δ ML−Fixed','Relative']))}</div><div>${panel('Noise evidence','Exact frozen values',table(n,['scope','graphs','cases','ml_mean','fixed_mean','oracle_mean','mean_regret','oracle_top3_accuracy'],['Scope','Graphs','Cases','ML','Fixed','Oracle','Regret','Oracle top-3']))}</div></div>`;
  chart('budgetChart',[traceMetric(b,'scope','ml_mean','ML'),traceMetric(b,'scope','fixed_mean','Fixed'),traceMetric(b,'scope','oracle_mean','Oracle')],{yaxis:{title:'Expected approximation ratio',range:[.66,.77]}});
  chart('noiseChart',[traceMetric(n,'scope','ml_mean','ML'),traceMetric(n,'scope','fixed_mean','Fixed'),traceMetric(n,'scope','oracle_mean','Oracle')],{yaxis:{title:'Expected approximation ratio',range:[.66,.77]}});
}
function traceMetric(rows,x,y,name){return{x:rows.map(r=>r[x]),y:rows.map(r=>+r[y]),type:'scatter',mode:'lines+markers',name,line:{width:2.5},marker:{size:8}}}
function table(rows,cols,labels,limit=100){return `<div class="table-wrap"><table class="data-table"><thead><tr>${labels.map(esc).map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,limit).map(r=>`<tr>${cols.map(c=>`<td>${formatCell(c,r[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
function formatCell(c,v){if(v===''||v==null)return'—';if(c.includes('percent'))return pct(v);if(['ml_mean','fixed_mean','random_expected_mean','oracle_mean','mean_regret','ml_minus_fixed','predicted_quality','selected_quality','oracle_quality','fixed_quality','random_expected_quality','regret','mae','rmse','r2'].includes(c))return fmt(v,5);if(c.includes('accuracy')||c==='ml_better_graph_fraction')return pct(v);return esc(v)}

function renderNoise(){
  const rows=currentBN();
  const matrix=rows.map(r=>({...r,label:`${r.budget||r.scope.split('_')[0]} / ${r.noise_condition||r.scope.split('_')[1]}`}));
  $('#page-noise').innerHTML=pageTitle('ROBUSTNESS / NOISE','Noise & Robustness','Read the selector under ideal, moderate-noise and higher-noise execution conditions. Values below are frozen artifacts.',seg('noise',['ALL','N0','N1','N2'],state.noise))+`
  <div class="grid three">${['N0','N1','N2'].map(id=>{const r=currentNoise().find(x=>x.scope===id)||{};return `<div class="panel"><div class="eyebrow">${id}</div><div class="stat-big">${fmt(r.ml_mean,5)}</div><div class="panel-sub">ML expected approximation ratio</div><div class="section-gap metric-pair"><div class="metric-box"><div class="label">Fixed</div><div class="value">${fmt(r.fixed_mean,5)}</div></div><div class="metric-box"><div class="label">Regret</div><div class="value">${fmt(r.mean_regret,5)}</div></div></div><div class="section-gap">${id==='N0'?'<span class="status ok">IDEAL SIMULATION</span>':id==='N1'?'<span class="status warn">MODERATE NOISE</span>':'<span class="status warn">HIGHER NOISE</span>'}</div></div>`}).join('')}</div>
  <div class="grid two section-gap"><div>${panel('Noise response curve','ML / fixed / oracle',`<div id="noiseCurve" class="chart tall"></div>`)}</div><div>${panel('Budget × noise matrix','ML selected performance',`<div id="noiseHeat" class="chart tall"></div>`)}</div></div>
  <div class="section-gap">${panel('Condition-level evidence','Every frozen budget/noise cell',table(matrix,['scope','graphs','cases','ml_mean','fixed_mean','oracle_mean','ml_minus_fixed','relative_gain_percent','mean_regret','oracle_top3_accuracy'],['Cell','Graphs','Cases','ML','Fixed','Oracle','Δ','Relative','Regret','Top-3']))}</div>`;
  chart('noiseCurve',[traceMetric(currentNoise(),'scope','ml_mean','ML'),traceMetric(currentNoise(),'scope','fixed_mean','Fixed'),traceMetric(currentNoise(),'scope','oracle_mean','Oracle')],{yaxis:{range:[.66,.77],title:'Expected approximation ratio'}});
  chart('noiseHeat',[{x:['N0','N1','N2'],y:['B256','B512'],z:[['N0','N1','N2'].map(n=>+(rows.find(r=>r.scope===`B256_${n}`)?.ml_mean||0)),['N0','N1','N2'].map(n=>+(rows.find(r=>r.scope===`B512_${n}`)?.ml_mean||0))],type:'heatmap',colorscale:[[0,'#121a29'],[.5,'#19556a'],[1,'#63e6ff']],hovertemplate:'%{y} / %{x}: %{z:.5f}<extra></extra>'}],{xaxis:{title:'Noise'},yaxis:{title:'Budget'}});
}

function renderBudget(){
  const rows=currentBN();
  $('#page-budget').innerHTML=pageTitle('ROBUSTNESS / RESOURCES','Budget Analysis','See how the selector behaves when the per-circuit shot budget changes from B256 to B512.')+`
  <div class="grid two"><div>${panel('Budget response','ML, fixed and oracle',`<div id="budgetResponse" class="chart tall"></div>`)}</div><div>${panel('Marginal ML gain','B512 − B256',`<div class="grid three">${['N0','N1','N2'].map(n=>{const a=+rows.find(r=>r.scope===`B256_${n}`)?.ml_mean||0,b=+rows.find(r=>r.scope===`B512_${n}`)?.ml_mean||0;return `<div class="metric-box"><div class="label">${n}</div><div class="value ${b-a>=0?'positive':'negative'}">${signed(b-a,5)}</div><div class="kpi-sub">ML ratio change</div></div>`}).join('')}</div><div class="section-gap callout"><strong>Read the trade-off:</strong> budget is not a monotonic guarantee of selector advantage. Inspect the noise-conditioned cells rather than treating B512 as universally better.</div>`)}</div></div>
  <div class="section-gap">${panel('Budget × noise evidence','Frozen six-cell experiment matrix',table(rows,['scope','ml_mean','fixed_mean','oracle_mean','ml_minus_fixed','relative_gain_percent','mean_regret'],['Cell','ML','Fixed','Oracle','Δ ML−Fixed','Relative','Regret']))}</div>`;
  chart('budgetResponse',[traceMetric(rows,'scope','ml_mean','ML'),traceMetric(rows,'scope','fixed_mean','Fixed'),traceMetric(rows,'scope','oracle_mean','Oracle')],{yaxis:{range:[.66,.77],title:'Expected approximation ratio'},xaxis:{tickangle:-20}});
}

function renderConfigs(){
  $('#page-configs').innerHTML=pageTitle('WORKSPACE / SEARCH SPACE','Configuration Lab','The canonical 12-configuration registry. Click any configuration to inspect its execution descriptors and observed behavior.')+`
  <div class="grid three"><div>${panel('Search-space contract','3 depths × 2 optimizers × 2 shot budgets',`<div class="stat-big">12</div><div class="panel-sub">stable configuration IDs</div>`)}</div><div>${panel('Depths','Circuit depth candidates',`<div class="hero-badges"><span class="badge">p = 1</span><span class="badge">p = 2</span><span class="badge">p = 3</span></div>`)}</div><div>${panel('Execution choices','Optimizer / shots',`<div class="hero-badges"><span class="badge cyan">COBYLA</span><span class="badge">SPSA</span><span class="badge">256 shots</span><span class="badge">512 shots</span></div>`)}</div></div>
  <div class="section-gap config-grid">${CONFIGS.map(c=>`<div class="config-card" data-config="${c.config_id}"><div class="config-card-head"><span class="config-id">${c.config_id}</span><span class="config-depth">DEPTH ${c.depth}</span></div><div class="config-main">${c.optimizer}</div><div class="config-meta"><span class="tag">${c.shots} shots</span><span class="tag">max evals 80</span></div></div>`).join('')}</div>`;
}
function openConfig(id){
  const c=CONFIGS.find(x=>x.config_id===id); if(!c)return;
  const rows=state.data.selected||[]; const observed=rows.filter(r=>r.selected_config_id===id); const avg=mean(observed.map(r=>r.selected_quality));
  openModal(`<button class="modal-close" data-close>×</button><div class="eyebrow">CONFIGURATION / ${id}</div><h1>${id} · depth ${c.depth}</h1><p class="panel-sub">Canonical configuration descriptor from the project registry.</p><div class="grid three section-gap"><div class="metric-box"><div class="label">Optimizer</div><div class="value">${c.optimizer}</div></div><div class="metric-box"><div class="label">Shots</div><div class="value">${c.shots}</div></div><div class="metric-box"><div class="label">Observed selections</div><div class="value">${observed.length}</div></div></div><div class="section-gap callout"><strong>Observed selected-quality mean:</strong> ${Number.isFinite(avg)?fmt(avg,5):'Not selected in the displayed test artifact.'}<br><br>This view is descriptive: selection frequency is not a causal importance measure.</div>`);
}

function renderGraphs(){
  const gs=graphMap(); const ids=selectedGraphs(); const features=state.data.features||[];
  const rows=ids.map(id=>{const f=features.find(x=>x.graph_id===id)||{};const g=gs[id]||{};return {graph_id:id,family:g.family||g.graph_family||'—',num_nodes:g.num_nodes||f.num_nodes||'—',num_edges:g.num_edges||'—',density:g.density||f.density||'—',average_degree:f.average_degree||g.degree_mean||'—'};});
  $('#page-graphs').innerHTML=pageTitle('WORKSPACE / GRAPH ANALYTICS','Graph Observatory','Inspect the held-out graph population, structural descriptors and selector behavior at graph level.',`<div class="global-search" style="width:210px"><span>⌕</span><input id="graphSearch" placeholder="Search G0001..." /></div>`)+`
  <div class="grid kpi-grid">${kpi('Test graphs',ids.length,'unique graph IDs')}${kpi('Families',new Set(rows.map(r=>r.family)).size,'graph families')}${kpi('Node range',`${Math.min(...rows.map(r=>+r.num_nodes||0))}–${Math.max(...rows.map(r=>+r.num_nodes||0))}`,'held-out population')}${kpi('Features',Object.keys(features[0]||{}).length,'graph descriptors')}${kpi('B256 cases',rows.length*3,'noise conditions')}${kpi('B512 cases',rows.length*3,'noise conditions')}</div>
  <div class="grid two"><div>${panel('Graph population','Node count distribution',`<div id="nodeHist" class="chart"></div>`)}</div><div>${panel('Graph density','Structural complexity',`<div id="densityScatter" class="chart"></div>`)}</div></div>
  <div class="section-gap">${panel('Held-out graph explorer','Click a graph for a full decision trace',`<div class="table-wrap"><table class="data-table"><thead><tr><th>Graph</th><th>Family</th><th>Nodes</th><th>Edges</th><th>Density</th><th>Avg degree</th><th>Cases</th></tr></thead><tbody>${rows.map(r=>`<tr class="clickable" data-graph="${r.graph_id}"><td><span class="config-chip">${r.graph_id}</span></td><td>${esc(r.family)}</td><td>${r.num_nodes}</td><td>${r.num_edges}</td><td>${fmt(r.density,4)}</td><td>${fmt(r.average_degree,3)}</td><td>6</td></tr>`).join('')}</tbody></table></div>`)}</div>`;
  const nums=rows.map(r=>+r.num_nodes); chart('nodeHist',[{x:nums,type:'histogram',marker:{color:'#63e6ff'},nbinsx:8,hovertemplate:'Nodes: %{x}<extra></extra>'}],{showlegend:false,xaxis:{title:'Number of nodes'},yaxis:{title:'Graphs'}});
  chart('densityScatter',[{x:rows.map(r=>+r.num_nodes),y:rows.map(r=>+r.density),text:rows.map(r=>r.graph_id),mode:'markers',type:'scatter',marker:{size:10,color:rows.map(r=>+r.density),colorscale:[[0,'#28384e'],[1,'#63e6ff']]},hovertemplate:'%{text}<br>nodes=%{x}<br>density=%{y:.4f}<extra></extra>'}],{showlegend:false,xaxis:{title:'Nodes'},yaxis:{title:'Density'}});
  $('#graphSearch')?.addEventListener('input',e=>{$$('.clickable').forEach(r=>r.style.display=r.dataset.graph.toLowerCase().includes(e.target.value.toLowerCase())?'':'none')});
}
function openGraph(id){
  const rows=(state.data.selected||[]).filter(r=>r.graph_id===id); const f=(state.data.features||[]).find(r=>r.graph_id===id)||{}; const g=(state.data.graphs||[]).find(r=>r.graph_id===id)||{};
  openModal(`<button class="modal-close" data-close>×</button><div class="eyebrow">GRAPH OBSERVATORY / ${id}</div><h1>${id}</h1><p class="panel-sub">Held-out graph decision trace from frozen test artifacts.</p><div class="grid three section-gap">${[['Family',g.family||g.graph_family||'—'],['Nodes',g.num_nodes||f.num_nodes||'—'],['Edges',g.num_edges||'—'],['Density',fmt(f.density||g.density,4)],['Avg degree',fmt(f.average_degree||g.degree_mean,3)],['Clustering',fmt(f.average_clustering,4)]].map(([l,v])=>`<div class="metric-box"><div class="label">${l}</div><div class="value">${esc(v)}</div></div>`).join('')}</div><div class="section-gap">${panel('Selector decisions','All available frozen rows for this graph',table(rows,['noise_condition','budget','selected_config_id','selected_quality','predicted_quality','oracle_config_id','oracle_quality','regret','selected_is_oracle','oracle_in_top3'],['Noise','Budget','Selected','Actual','Predicted','Oracle','Oracle quality','Regret','Oracle?','Top-3']))}</div>`);
}

function renderSelector(){
  const v=(state.data.validation||[]).find(r=>r.model==='extra_trees')||{}; const m=currentMain();
  $('#page-selector').innerHTML=pageTitle('ML INTELLIGENCE / SELECTOR','ML Selector','Trace how graph/context information becomes a configuration ranking. The displayed model metrics come from the frozen validation artifact.')+`
  <div class="grid kpi-grid">${kpi('Model','Extra Trees','frozen validation model','accent')}${kpi('Validation cases',v.cases||78,'graph-level cases')}${kpi('MAE',fmt(v.mae,5),'quality prediction')}${kpi('RMSE',fmt(v.rmse,5),'quality prediction')}${kpi('R²',fmt(v.r2,5),'held-out validation')}${kpi('Mean regret',fmt(v.mean_regret,5),'selected vs oracle')}</div>
  <div class="panel"><div class="eyebrow">DECISION PIPELINE</div><div class="timeline"><div class="step"><b>01 · Graph</b><small>Topology, size, density and spectral descriptors.</small></div><div class="step"><b>02 · Context</b><small>Noise condition and configuration descriptors.</small></div><div class="step"><b>03 · Prediction</b><small>Extra Trees estimates configuration quality.</small></div><div class="step"><b>04 · Selection</b><small>Highest predicted feasible quality is selected.</small></div></div></div>
  <div class="grid two section-gap"><div>${panel('Model quality','Validation comparison',`<div id="modelChart" class="chart"></div>`)}</div><div>${panel('Selector vs oracle','Validation quality',`<div class="metric-pair"><div class="metric-box"><div class="label">Selection mean</div><div class="value">${fmt(v.selection_mean,5)}</div></div><div class="metric-box"><div class="label">Oracle mean</div><div class="value">${fmt(v.oracle_mean,5)}</div></div></div><div class="section-gap callout"><strong>Oracle selection accuracy:</strong> ${pct(v.oracle_selection_accuracy)}<br><strong>Oracle top-3:</strong> ${pct(v.oracle_top3_accuracy)}<br><br>These metrics describe ranking behavior, not a claim that the selector exactly reconstructs the oracle.</div>`)}</div></div>`;
  const val=state.data.validation||[]; chart('modelChart',[{x:val.map(r=>r.model.replaceAll('_',' ')),y:val.map(r=>+r.mae),type:'bar',name:'MAE',marker:{color:'#63e6ff'}},{x:val.map(r=>r.model.replaceAll('_',' ')),y:val.map(r=>+r.rmse),type:'bar',name:'RMSE',marker:{color:'#9c7cff'}}],{barmode:'group',yaxis:{title:'Error',tickformat:'.03f'}});
}

function renderFeatures(){
  const rows=state.data.importance||[]; const max=Math.max(...rows.map(r=>+r.mean_regret_increase||0),.0001);
  $('#page-features').innerHTML=pageTitle('ML INTELLIGENCE / EXPLAINABILITY','Feature Intelligence','Permutation importance measured as increase in validation regret when a feature group is permuted. Higher values indicate greater sensitivity in this analysis.')+`
  <div class="grid two"><div>${panel('Permutation importance','30 repetitions per feature group',`<div class="feature-list">${rows.slice(0,18).map(r=>`<div class="feature-item"><div class="feature-name">${esc(r.feature_group)}</div><div class="feature-bar"><span style="width:${(+r.mean_regret_increase/max*100).toFixed(1)}%"></span></div><div class="feature-num">${fmt(r.mean_regret_increase,5)}</div></div>`).join('')}</div>`)}</div><div>${panel('Interpretation','Validation regret sensitivity',`<div id="importanceChart" class="chart tall"></div><div class="section-gap research-note">Permutation importance is model-dependent and should be read as an explanatory diagnostic, not a causal graph-theoretic statement.</div>`)}</div></div>`;
  chart('importanceChart',[{x:rows.slice(0,12).map(r=>r.feature_group),y:rows.slice(0,12).map(r=>+r.mean_regret_increase),type:'bar',orientation:'h',marker:{color:'#9c7cff'},hovertemplate:'%{y}: %{x:.5f}<extra></extra>'}],{margin:{l:180,r:20,t:12,b:35},xaxis:{title:'Mean regret increase'},yaxis:{autorange:'reversed'}});
}

function renderAblation(){
  const rows=state.data.ablation||[];
  $('#page-ablation').innerHTML=pageTitle('ML INTELLIGENCE / ABLATION','Ablation Lab','Remove feature groups from the selector and observe how validation prediction and selection behavior changes.')+`
  <div class="grid three">${rows.map(r=>`<div class="panel"><div class="eyebrow">${esc(r.ablation)}</div><div class="stat-big">${fmt(r.mean_regret,5)}</div><div class="panel-sub">mean regret</div><div class="section-gap metric-pair"><div class="metric-box"><div class="label">MAE</div><div class="value">${fmt(r.mae,5)}</div></div><div class="metric-box"><div class="label">R²</div><div class="value">${fmt(r.r2,4)}</div></div></div><div class="section-gap">Top-3: <span class="positive">${pct(r.oracle_top3_accuracy)}</span></div></div>`).join('')}</div>
  <div class="section-gap">${panel('Ablation comparison','Lower regret is closer to the oracle',`<div id="ablationChart" class="chart"></div>`)}</div>`;
  chart('ablationChart',[{x:rows.map(r=>r.ablation.replace('without_','− ')),y:rows.map(r=>+r.mean_regret),type:'bar',marker:{color:'#63e6ff'},hovertemplate:'%{x}<br>regret=%{y:.5f}<extra></extra>'}],{showlegend:false,yaxis:{title:'Mean regret'}});
}

function renderValidation(){
  const rows=state.data.validation||[];
  $('#page-validation').innerHTML=pageTitle('RESEARCH / VALIDATION','Model Validation','Compare the candidate regressors on the graph-level validation population before the frozen test evaluation.')+`
  <div class="grid two"><div>${panel('Validation model matrix','Frozen metrics',table(rows,['model','mae','rmse','r2','selection_mean','oracle_mean','mean_regret','median_regret','oracle_selection_accuracy','oracle_top3_accuracy','cases'],['Model','MAE','RMSE','R²','Selection','Oracle','Regret','Median regret','Oracle acc.','Top-3','Cases']))}</div><div>${panel('Error profile','MAE vs RMSE',`<div id="validationChart" class="chart tall"></div>`)}</div></div>`;
  chart('validationChart',[{x:rows.map(r=>r.model.replaceAll('_',' ')),y:rows.map(r=>+r.mae),type:'bar',name:'MAE',marker:{color:'#63e6ff'}},{x:rows.map(r=>r.model.replaceAll('_',' ')),y:rows.map(r=>+r.rmse),type:'bar',name:'RMSE',marker:{color:'#9c7cff'}}],{barmode:'group',yaxis:{title:'Prediction error'}});
}

function renderExplorer(){
  const rows=filteredSelected();
  $('#page-explorer').innerHTML=pageTitle('WORKSPACE / CASE EXPLORER','Experiment Explorer','Search and inspect the frozen per-graph selector decisions. Filters affect the visible table only; no research artifact is modified.',`${seg('explorerBudget',['ALL','B256','B512'],state.budget)} ${seg('explorerNoise',['ALL','N0','N1','N2'],state.noise)}`)+`
  <div class="grid kpi-grid">${kpi('Visible cases',rows.length,'filtered rows')}${kpi('Graphs',new Set(rows.map(r=>r.graph_id)).size,'unique graphs')}${kpi('Oracle matches',rows.filter(r=>r.selected_is_oracle==='True'||r.selected_is_oracle==='true').length,'selected = oracle')}${kpi('Top-3 coverage',pct(mean(rows.map(r=>String(r.oracle_in_top3).toLowerCase()==='true'?1:0))),'oracle contained')}${kpi('Mean selected quality',fmt(mean(rows.map(r=>+r.selected_quality)),5),'visible cases')}${kpi('Mean regret',fmt(mean(rows.map(r=>+r.regret)),5),'visible cases')}</div>
  <div class="section-gap">${panel('Decision table','Click a row for the complete configuration decision trace',`<div class="table-wrap"><table class="data-table"><thead><tr><th>Graph</th><th>Noise</th><th>Budget</th><th>Selected</th><th>Selected quality</th><th>Predicted</th><th>Oracle</th><th>Regret</th><th>Oracle</th><th>Top-3</th></tr></thead><tbody>${rows.slice(0,120).map(r=>`<tr class="clickable" data-case="${r.graph_id}|${r.noise_condition}|${r.budget}"><td><span class="config-chip">${r.graph_id}</span></td><td>${r.noise_condition}</td><td>${r.budget}</td><td>${r.selected_config_id}</td><td>${fmt(r.selected_quality,5)}</td><td>${fmt(r.predicted_quality,5)}</td><td>${r.oracle_config_id}</td><td class="${+r.regret===0?'positive':''}">${fmt(r.regret,5)}</td><td>${rowStatus(String(r.selected_is_oracle).toLowerCase()==='true')}</td><td>${rowStatus(String(r.oracle_in_top3).toLowerCase()==='true')}</td></tr>`).join('')}</tbody></table></div>`)}</div>`;
}
function openCase(key){const [g,n,b]=key.split('|');const r=(state.data.selected||[]).find(x=>x.graph_id===g&&x.noise_condition===n&&x.budget===b);if(!r)return;openModal(`<button class="modal-close" data-close>×</button><div class="eyebrow">CASE TRACE</div><h1>${g} · ${n} · ${b}</h1><p class="panel-sub">Frozen selector decision for one graph / environment / resource-budget case.</p><div class="grid three section-gap">${[['Selected',r.selected_config_id],['Predicted quality',fmt(r.predicted_quality,5)],['Observed quality',fmt(r.selected_quality,5)],['Oracle',r.oracle_config_id],['Oracle quality',fmt(r.oracle_quality,5)],['Regret',fmt(r.regret,5)]].map(([l,v])=>`<div class="metric-box"><div class="label">${l}</div><div class="value">${esc(v)}</div></div>`).join('')}</div><div class="section-gap callout"><strong>Decision state:</strong> ${String(r.selected_is_oracle).toLowerCase()==='true'?'selected configuration matches the oracle for this case.':'selected configuration differs from the oracle for this case.'}<br><strong>Oracle top-3:</strong> ${String(r.oracle_in_top3).toLowerCase()==='true'?'yes':'no'}<br><strong>Fixed baseline:</strong> ${r.fixed_config_id} · ${fmt(r.fixed_quality,5)}<br><strong>Random expectation:</strong> ${fmt(r.random_expected_quality,5)}</div>`)}

function renderStats(){
  const rows=state.data.stats||[]; const all=rows.find(r=>r.scope==='ALL')||{};
  $('#page-stats').innerHTML=pageTitle('RESEARCH / INFERENCE','Statistical Evidence','Graph-clustered comparison of ML-selected versus fixed-baseline performance. Descriptive differences and uncertainty are shown together.')+`
  <div class="grid three"><div class="panel"><div class="eyebrow">MEAN DIFFERENCE</div><div class="stat-big ${+all.mean_ml_minus_fixed>=0?'positive':'negative'}">${signed(all.mean_ml_minus_fixed,6)}</div><div class="panel-sub">ML − fixed</div></div><div class="panel"><div class="eyebrow">BOOTSTRAP 95% CI</div><div class="stat-big">[${fmt(all.bootstrap_ci95_low,4)}, ${fmt(all.bootstrap_ci95_high,4)}]</div><div class="panel-sub">graph-clustered bootstrap interval</div></div><div class="panel"><div class="eyebrow">SIGN-FLIP P-VALUE</div><div class="stat-big">${fmt(all.exact_sign_flip_pvalue,4)}</div><div class="panel-sub">exact test in frozen analysis</div></div></div>
  <div class="grid two section-gap"><div>${panel('Effect with uncertainty','The zero line is shown for orientation',`<div id="ciChart" class="chart tall"></div>`)}</div><div>${panel('Research reading',null,`<div class="callout"><strong>Overall:</strong> the frozen aggregate difference is ${signed(all.mean_ml_minus_fixed,6)} with a 95% bootstrap interval from ${fmt(all.bootstrap_ci95_low,5)} to ${fmt(all.bootstrap_ci95_high,5)} and exact sign-flip p-value ${fmt(all.exact_sign_flip_pvalue,4)}.</div><div class="section-gap research-note">The appropriate interpretation is evidence reporting, not a binary claim based on the point estimate alone. Subgroup cells are included for transparency and should not be over-read when sample sizes are small.</div>`)}</div></div>
  <div class="section-gap">${panel('All statistical scopes','Frozen test, budget, noise and budget × noise analyses',table(rows,['scope','graphs','mean_ml_minus_fixed','median_ml_minus_fixed','bootstrap_ci95_low','bootstrap_ci95_high','exact_sign_flip_pvalue','ml_better_graph_fraction','holm_adjusted_pvalue'],['Scope','Graphs','Mean Δ','Median Δ','CI low','CI high','P value','Graphs improved','Holm adj.']))}</div>`;
  const subset=rows.slice(0,12); chart('ciChart',[{x:subset.map(r=>r.scope),y:subset.map(r=>+r.mean_ml_minus_fixed),error_y:{type:'data',symmetric:false,array:subset.map(r=>Math.max(0,+r.bootstrap_ci95_high-+r.mean_ml_minus_fixed)),arrayminus:subset.map(r=>Math.max(0,+r.mean_ml_minus_fixed-+r.bootstrap_ci95_low))},type:'scatter',mode:'markers',marker:{size:9,color:'#63e6ff'},hovertemplate:'%{x}<br>Δ=%{y:.5f}<extra></extra>'},{x:subset.map(()=>''),y:subset.map(()=>0),type:'scatter',mode:'lines',line:{color:'#42536a',dash:'dash'},showlegend:false}],{yaxis:{title:'ML − fixed',zeroline:true,zerolinecolor:'#536176'}});
}

function renderRepro(){
  $('#page-repro').innerHTML=pageTitle('RESEARCH / PROVENANCE','Reproducibility Center','A compact audit surface for the frozen experiment contract, data split and artifact lineage.')+`
  <div class="grid kpi-grid">${kpi('Experiment seed','2027','project seed','accent')}${kpi('Graphs generated','100','primary population')}${kpi('Admitted graphs','89','complete N0/N1/config coverage')}${kpi('Train / val / test','62 / 13 / 14','graph-level split')}${kpi('QAOA configs','12','canonical registry')}${kpi('Noise / budgets','3 × 2','N0/N1/N2 · B256/B512')}</div>
  <div class="grid two"><div>${panel('Experiment contract','Authoritative configuration values',`<div class="table-wrap"><table class="data-table"><tbody>${[['Problem','MaxCut'],['Graph families','Erdos–Renyi · Random Regular'],['Node counts','10 · 12 · 15 · 18 · 20'],['ER probability','0.35'],['Random regular degree','4'],['Depths','1 · 2 · 3'],['Optimizers','COBYLA · SPSA'],['Shots','256 · 512'],['Max optimizer evaluations','80'],['Exact solver max nodes','20'],['Primary alpha','0.8'],['Tie tolerance','0.005']].map(([a,b])=>`<tr><td>${a}</td><td class="mono">${b}</td></tr>`).join('')}</tbody></table></div>`)}</div><div>${panel('Artifact lineage','Committed research outputs',`<div class="feature-list">${[['T1','Main performance'],['T2','Budget analysis'],['T3','Noise analysis'],['T4','Budget × noise'],['T5','Model validation'],['T6','Feature ablation'],['T7','Permutation importance'],['T8','Statistical tests'],['T9','Selected configurations'],['T10','Fixed baselines'],['T11','Graph features'],['T12','Experiment metadata']].map(([a,b])=>`<div class="comparison-row"><div class="comparison-label"><span class="config-chip">${a}</span></div><div class="bar-track"><div class="bar-fill" style="width:100%"></div></div><div class="comparison-val">${b}</div></div>`).join('')}</div>`)}</div></div>
  <div class="section-gap callout"><strong>Important:</strong> this dashboard reads the repository artifacts directly. If a result is changed in the experiment, the corresponding committed artifact and dashboard view should be regenerated together; the UI should never become a second source of truth.</div>`;
}

function renderMethod(){
  $('#page-method').innerHTML=pageTitle('RESEARCH / METHODS','Methodology','A plain-language map of the experiment so a reviewer can move from problem definition to evaluation without hunting through scripts.')+`
  <div class="grid two"><div>${panel('01 · Problem','Combinatorial optimization',`<h2>MaxCut</h2><p class="panel-sub" style="font-size:11px;line-height:1.8">Given a graph, find a partition of vertices that maximizes the number of crossing edges. QAOA supplies a parameterized quantum circuit family; the selector decides which tested configuration to allocate the limited execution budget to.</p>`)}</div><div>${panel('02 · Search space','Canonical QAOA registry',`<div class="config-grid">${CONFIGS.map(c=>`<div class="config-card"><div class="config-id">${c.config_id}</div><div class="panel-sub">p=${c.depth} · ${c.optimizer}</div><div class="panel-sub">${c.shots} shots</div></div>`).join('')}</div>`)}</div></div>
  <div class="section-gap panel"><div class="eyebrow">03 · SELECTION LOGIC</div><div class="timeline"><div class="step"><b>Generate graph features</b><small>Topology, spectral and connectivity descriptors are derived from the graph.</small></div><div class="step"><b>Attach context</b><small>Noise condition and configuration descriptors define the prediction context.</small></div><div class="step"><b>Predict quality</b><small>The trained selector estimates expected configuration quality.</small></div><div class="step"><b>Rank and execute</b><small>The highest predicted feasible configuration is selected under the resource budget.</small></div></div></div>
  <div class="grid three section-gap"><div>${panel('Baseline','Fixed configuration',`<p class="panel-sub">A task-aligned fixed configuration is chosen separately for each budget/noise cell from training data and then frozen for test comparison.</p>`)}</div><div>${panel('Oracle','Reference upper envelope',`<p class="panel-sub">The oracle uses the best observed configuration for the evaluated graph/case. It is a reference, not a deployable selector.</p>`)}</div><div>${panel('Random','Chance expectation',`<p class="panel-sub">Random configuration expectation provides a lower reference for contextual selection performance.</p>`)}</div></div>`;
}

function renderPage(){
  const pages=['overview','explorer','performance','graphs','configs','noise','budget','selector','features','ablation','validation','stats','repro','method'];
  pages.forEach(p=>document.getElementById(`page-${p}`).classList.toggle('hidden',state.page!==p));
  const names={overview:'Overview',explorer:'Experiment Explorer',performance:'Performance',graphs:'Graph Observatory',configs:'Configuration Lab',noise:'Noise & Robustness',budget:'Budget Analysis',selector:'ML Selector',features:'Feature Intelligence',ablation:'Ablation Lab',validation:'Model Validation',stats:'Statistical Evidence',repro:'Reproducibility',method:'Methodology'};
  $('#pageCrumb').textContent=names[state.page];
  $$('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.page===state.page));
  const fn={overview:renderOverview,explorer:renderExplorer,performance:renderPerformance,graphs:renderGraphs,configs:renderConfigs,noise:renderNoise,budget:renderBudget,selector:renderSelector,features:renderFeatures,ablation:renderAblation,validation:renderValidation,stats:renderStats,repro:renderRepro,method:renderMethod}[state.page];
  try{fn?.();bindPageEvents();}catch(e){console.error(e);document.getElementById(`page-${state.page}`).innerHTML=`<div class="error"><strong>UI rendering error.</strong><br>${esc(e.message)}</div>`;}
}

function bindPageEvents(){
  $$('.nav-item').forEach(b=>b.onclick=()=>{state.page=b.dataset.page;$('#sidebar').classList.remove('open');renderPage();window.scrollTo({top:0,behavior:'smooth'});});
  $$('[data-seg]').forEach(s=>$$('button',s).forEach(b=>b.onclick=()=>{
    const v=b.dataset.value;
    if(s.dataset.seg.toLowerCase().includes('budget'))state.budget=v;
    if(s.dataset.seg.toLowerCase().includes('noise'))state.noise=v;
    renderPage();
  }));
  $$('[data-config]').forEach(x=>x.onclick=()=>openConfig(x.dataset.config));
  $$('[data-graph]').forEach(x=>x.onclick=()=>openGraph(x.dataset.graph));
  $$('[data-case]').forEach(x=>x.onclick=()=>openCase(x.dataset.case));
  $$('[data-close]').forEach(x=>x.onclick=closeModal);
}
function openModal(html){$('#modal').innerHTML=html;$('#modalBackdrop').classList.remove('hidden');$$('[data-close]').forEach(x=>x.onclick=closeModal)}
function closeModal(){ $('#modalBackdrop').classList.add('hidden'); }
function toast(msg){const t=$('#toast');t.textContent=msg;t.classList.add('show');clearTimeout(window.__toast);window.__toast=setTimeout(()=>t.classList.remove('show'),2200)}

function globalSearch(q){
  const term=q.trim().toLowerCase(); if(!term)return;
  const pages=[['overview','Overview'],['explorer','Experiment Explorer'],['performance','Performance'],['graphs','Graph Observatory'],['configs','Configuration Lab'],['noise','Noise & Robustness'],['budget','Budget Analysis'],['selector','ML Selector'],['features','Feature Intelligence'],['ablation','Ablation Lab'],['validation','Model Validation'],['stats','Statistical Evidence'],['repro','Reproducibility'],['method','Methodology']];
  const page=pages.find(p=>p[1].toLowerCase().includes(term)); if(page){state.page=page[0];renderPage();return;}
  const graph=selectedGraphs().find(id=>id.toLowerCase()===term);if(graph){state.page='graphs';renderPage();setTimeout(()=>openGraph(graph),80);return;}
  const config=CONFIGS.find(c=>c.config_id.toLowerCase()===term);if(config){state.page='configs';renderPage();setTimeout(()=>openConfig(config.config_id),80);return;}
  toast('No matching page, graph or configuration found.');
}

$('#menuBtn')?.addEventListener('click',()=>$('#sidebar').classList.toggle('open'));
$('#themeBtn')?.addEventListener('click',()=>{state.theme=state.theme==='dark'?'contrast':'dark';document.documentElement.style.setProperty('--bg',state.theme==='contrast'?'#05070c':'#070a12');toast(state.theme==='contrast'?'High-contrast mode enabled':'Standard dark mode enabled')});
$('#refreshBtn')?.addEventListener('click',async()=>{toast('Refreshing frozen artifacts…');await boot();toast('Research artifacts refreshed')});
$('#globalSearch')?.addEventListener('keydown',e=>{if(e.key==='Enter')globalSearch(e.target.value)});
$('#modalBackdrop')?.addEventListener('click',e=>{if(e.target.id==='modalBackdrop')closeModal()});

async function boot(){
  const first=$('#page-overview');first.innerHTML='<div class="loading"><div><div class="spinner"></div><div style="margin-top:12px;font:500 9px DM Mono,monospace">LOADING FROZEN RESEARCH ARTIFACTS</div></div></div>';
  try{await loadAll();renderPage();}catch(e){first.innerHTML=`<div class="error">Unable to load dashboard artifacts: ${esc(e.message)}</div>`;}
}
boot();
