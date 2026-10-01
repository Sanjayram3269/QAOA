import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const root = path.resolve(process.cwd(), '..');
const dashboard = path.resolve(process.cwd());

function csv(text){
  const lines=text.trim().split(/\r?\n/);
  const headers=lines.shift().split(',');
  return lines.map(line=>{
    const cells=[];let cell='',q=false;
    for(let i=0;i<line.length;i++){
      const c=line[i],n=line[i+1];
      if(c==='"'){if(q&&n==='"'){cell+='"';i++;}else q=!q;}
      else if(c===','&&!q){cells.push(cell);cell='';} else cell+=c;
    }
    cells.push(cell);return Object.fromEntries(headers.map((h,i)=>[h,cells[i]??'']));
  });
}
async function read(rel){return fs.readFile(path.join(root,rel),'utf8');}

const required=[
  'data/ml/paper_results/01_main_results.csv',
  'data/ml/paper_results/02_budget_results.csv',
  'data/ml/paper_results/03_noise_results.csv',
  'data/ml/paper_results/04_budget_noise_results.csv',
  'data/ml/paper_results/05_model_validation_comparison.csv',
  'data/ml/paper_results/06_feature_ablation.csv',
  'data/ml/paper_results/07_permutation_importance.csv',
  'data/ml/paper_results/08_statistical_tests.csv',
  'data/ml/paper_results/09_test_selected_configurations.csv',
  'data/ml/paper_results/10_fixed_baseline_choices.csv',
  'data/ml/paper_results/11_test_graph_features.csv',
  'data/ml/paper_results/12_experiment_metadata.csv',
  'data/graphs/master_graph_manifest.csv',
  'config/experiment.yaml'
];
for(const f of required){await fs.access(path.join(root,f));}

const main=csv(await read('data/ml/paper_results/01_main_results.csv'))[0];
assert.equal(main.scope,'ALL');
assert.equal(Number(main.graphs),14);
assert.equal(Number(main.cases),84);
assert(Math.abs(Number(main.ml_mean)-0.7022283078825987)<1e-12);
assert(Math.abs(Number(main.fixed_mean)-0.7009595877315616)<1e-12);
assert(Math.abs(Number(main.oracle_mean)-0.7243513088582523)<1e-12);

const budget=csv(await read('data/ml/paper_results/02_budget_results.csv'));
assert.deepEqual(budget.map(x=>x.scope),['B256','B512']);
assert(budget.every(x=>Number(x.graphs)===14));

const noise=csv(await read('data/ml/paper_results/03_noise_results.csv'));
assert.deepEqual(noise.map(x=>x.scope),['N0','N1','N2']);

const cells=csv(await read('data/ml/paper_results/04_budget_noise_results.csv'));
assert.equal(cells.length,6);
assert.deepEqual(cells.map(x=>x.scope),['B256_N0','B256_N1','B256_N2','B512_N0','B512_N1','B512_N2']);

const selected=csv(await read('data/ml/paper_results/09_test_selected_configurations.csv'));
assert.equal(selected.length,84);
assert(selected.every(x=>/^C(0[1-9]|1[0-2])$/.test(x.selected_config_id)));
assert(selected.every(x=>['N0','N1','N2'].includes(x.noise_condition)));
assert(selected.every(x=>['B256','B512'].includes(x.budget)));

const features=csv(await read('data/ml/paper_results/11_test_graph_features.csv'));
assert(features.length>=14,'Expected held-out graph feature rows');
assert(features[0].graph_id,'Graph feature rows must have graph IDs');

const validation=csv(await read('data/ml/paper_results/05_model_validation_comparison.csv'));
assert(validation.some(x=>x.model==='extra_trees'),'Extra Trees validation row missing');

const ablation=csv(await read('data/ml/paper_results/06_feature_ablation.csv'));
assert(ablation.some(x=>x.ablation==='full'),'Full-feature ablation row missing');

const importance=csv(await read('data/ml/paper_results/07_permutation_importance.csv'));
assert(importance.length>=10,'Permutation importance artifact unexpectedly small');

const stats=csv(await read('data/ml/paper_results/08_statistical_tests.csv'));
const all=stats.find(x=>x.scope==='ALL');
assert(all,'Overall statistical row missing');
assert(Math.abs(Number(all.exact_sign_flip_pvalue)-0.809814453125)<1e-12);

const html=await fs.readFile(path.join(dashboard,'index.html'),'utf8');
const css=await fs.readFile(path.join(dashboard,'styles.css'),'utf8');
const app=await fs.readFile(path.join(dashboard,'app.js'),'utf8');
assert(html.includes('./styles.css'));
assert(html.includes('./app.js'));
assert(css.includes('--text:#f4f7fb'),'High-contrast text token missing');
assert(app.includes("01_main_results.csv"),'Dashboard must bind to real main result artifact');
assert(app.includes("09_test_selected_configurations.csv"),'Dashboard must bind to real decision artifact');

console.log(`Dashboard data contract OK: ${selected.length} cases, ${features.length} graph-feature rows, ${validation.length} models, ${importance.length} importance rows.`);
