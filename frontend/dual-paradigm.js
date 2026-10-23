// Inlined inside the workbench closure; shares state and existing safe renderers.
const experiments = Array.isArray(bundle.experiments) ? bundle.experiments : [];
let selectedExperiment = experiments.find(item => item.id === 'lora_full') || experiments[0];
const percent = value => typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(2)}%` : '—';

function renderOverview() {
  const models = bundle.metrics?.models || {};
  const full = experiments.find(item => item.id === 'lora_full');
  const cards = [
    ['BERT · 抽取式', percent(models.bert?.best_f1), '开发集 Micro F1 · 全量微调'],
    ['MacBERT · 抽取式', percent(models.macbert?.best_f1), '开发集 Micro F1 · 全量微调'],
    ['Qwen + LoRA · 生成式', percent(full?.metrics?.micro_f1), full ? `${full.metrics.n_sentences} 句开发集 · 独立训练配置` : '尚无全量实验记录'],
    ['统一任务 / CLUENER2020', '10 类实体', '严格实体匹配 · 边界与类别同时正确']
  ];
  document.getElementById('overviewStats').innerHTML = cards.map(([label, value, note]) => `<div class="overview-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></div>`).join('');
  document.getElementById('overviewTaxonomy').innerHTML = Object.entries(bundle.category_meta || {}).map(([key, meta]) => `<span class="entity-span badge-${escapeHtml(key)}">${escapeHtml(meta.cn || key)}</span>`).join('');
}

function renderGeneration() {
  document.getElementById('experimentList').innerHTML = experiments.map((item, index) => `<button class="experiment-option" data-experiment="${index}" aria-pressed="${item === selectedExperiment}"><strong>${escapeHtml(item.label)}</strong><small>${item.metrics.n_sentences} 句 · ${item.config.k}-shot · ${percent(item.metrics.micro_f1)}</small></button>`).join('');
  const exportButton = document.getElementById('exportExperiment');
  exportButton.disabled = !selectedExperiment;
  if (!selectedExperiment) {
    document.getElementById('experimentStats').innerHTML = '<p class="empty-state">暂无实验数据。添加实验结果后重新构建页面即可查看。</p>';
    document.getElementById('generationSource').textContent = '没有可用的原始实验文件；不以旧汇总数据填充指标。';
    return;
  }
  const {metrics: m, config: c, label, source} = selectedExperiment;
  document.getElementById('experimentTitle').textContent = label;
  document.getElementById('experimentScope').textContent = `${m.n_sentences} 句 · ${m.n_queries} 次类别查询 · ${m.n_gold_entities} 个标注实体`;
  document.getElementById('experimentStats').innerHTML = [
    ['Micro F1', m.micro_f1], ['Precision', m.micro_p], ['Recall', m.micro_r],
    ['格式异常率', typeof m.malformed_rate === 'number' ? m.malformed_rate * 100 : null],
    ['幻觉率', typeof m.hallucination_rate === 'number' ? m.hallucination_rate * 100 : null]
  ].map(([name, value]) => `<div class="experiment-stat"><span>${name}</span><strong>${percent(value)}</strong></div>`).join('');
  document.getElementById('experimentConfig').textContent = `检索: ${c.strategy}  /  k: ${c.k}  /  seed: ${c.seed}  /  max_new_tokens: ${c.max_new_tokens}  /  自验证: ${c.self_verify ? '开启' : '关闭'}`;
  document.getElementById('generationSource').textContent = `来源：${source}。格式异常率与幻觉率按类别查询统计；P / R / F1 为实体级指标。各运行的训练量和解码配置不同，请结合实验条件解读。`;
  renderGenerationBars();
}

function renderGenerationBars() {
  const metric = document.getElementById('generationMetric').value;
  const rows = Object.entries(selectedExperiment?.metrics?.per_type || {});
  document.getElementById('generationBars').innerHTML = rows.length ? rows.map(([key, item]) => {
    const value = item[metric];
    const width = typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
    return `<div class="generation-bar"><span>${escapeHtml(bundle.category_meta?.[key]?.cn || key)}</span><div aria-hidden="true"><i style="width:${width}%"></i></div><strong>${percent(value)}</strong><small>n = ${escapeHtml(String(item.support ?? '—'))}</small></div>`;
  }).join('') : '<p class="empty-state">这次运行没有逐类别指标。</p>';
}

function renderRetrievalAblation() {
  const rows = experiments.filter(item => item.group === 'retrieval');
  document.getElementById('retrievalAblation').innerHTML = rows.length ? rows.map(({label, metrics: m}) => `<tr><td>${escapeHtml(label)}</td><td>${percent(m.micro_p)}</td><td>${percent(m.micro_r)}</td><td><strong>${percent(m.micro_f1)}</strong></td><td>${percent(m.malformed_rate * 100)}</td><td>${percent(m.hallucination_rate * 100)}</td></tr>`).join('') : '<tr><td colspan="6">暂无同口径检索消融记录。</td></tr>';
}

function renderProtocol() {
  const category = document.getElementById('protocolCategory').value;
  const text = '张三在北京加入了腾讯。';
  const example = {name: {text: '张三', start: 0}, address: {text: '北京', start: 3}, company: {text: '腾讯', start: 8}}[category];
  const end = example.start + example.text.length - 1;
  document.getElementById('protocolPrompt').textContent = `请识别句子中的${bundle.category_meta?.[category]?.cn || category}，保持原文不变，只用 @@ 和 ## 包裹目标实体。\n输入：${text}`;
  document.getElementById('protocolOutput').textContent = `${text.slice(0, example.start)}@@${example.text}##${text.slice(end + 1)}`;
  document.getElementById('protocolEntities').textContent = JSON.stringify([{text: example.text, category, start: example.start, end}], null, 2);
}

function renderGenerativeCase(c) {
  return `<article class="case-card"><div class="case-top-meta"><span class="case-id-badge">${escapeHtml(c.case_id)}</span><span class="stat-chip">生成式 · 案例记录</span></div><h3 class="case-title">${escapeHtml(c.pattern_name)}</h3><p class="muted small">${escapeHtml(c.model_type)}</p><div class="case-text-box">${escapeHtml(c.text)}</div><div class="comparison-box"><div class="comp-row"><span class="comp-label">真实标注</span><div class="comp-entities">${renderEntitiesInlineBadges(c.ground_truth)}</div></div><div class="comp-row"><span class="comp-label">记录的预测</span><div class="comp-entities">${renderEntitiesInlineBadges(c.generative_prediction)}</div></div></div><details><summary class="small">查看原始生成文本</summary><pre class="generation-raw">${escapeHtml(c.model_output || '未记录原始输出')}</pre></details><div class="case-reason"><strong>现象分析：</strong>${escapeHtml(c.diff_analysis || '')}</div><details class="small"><summary>机制分析与改进方向</summary><p class="case-reason">${escapeHtml(c.technical_reason || '')}</p><p class="case-reason">${escapeHtml(c.proposed_solution || '')}</p></details><p class="source-note">来源：case_study_results.json · 历史案例注释，文中比例对应原运行，非当前选中实验。</p></article>`;
}

function syncBenchmarkSummary() {
  const models = bundle.metrics?.models || {};
  const m = models.macbert || {}, b = models.bert || {};
  document.getElementById('kpiMacbertF1').textContent = percent(m.best_f1);
  document.getElementById('kpiBertF1').textContent = percent(b.best_f1);
  const delta = typeof m.best_f1 === 'number' && typeof b.best_f1 === 'number' ? m.best_f1 - b.best_f1 : null;
  document.getElementById('benchmarkDelta').textContent = delta === null ? '—' : `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}pp`;
  document.getElementById('benchmarkDeltaNote').textContent = `相对 BERT (${percent(b.best_f1)}) 的本次结果差值；单次运行不代表统计显著。`;
  document.getElementById('benchmarkLoss').textContent = typeof m.val_loss === 'number' ? m.val_loss.toFixed(4) : '—';
  document.getElementById('benchmarkLossNote').textContent = `MacBERT / BERT 验证损失；BERT: ${typeof b.val_loss === 'number' ? b.val_loss.toFixed(4) : '—'}`;
  document.getElementById('benchmarkEpoch').textContent = `Epoch ${m.best_epoch ?? '—'}`;
  document.getElementById('benchmarkEpochNote').textContent = `本次最优轮次 · BERT: ${b.best_epoch ?? '—'}`;
  document.getElementById('trainingRunConfig').innerHTML = Object.entries(bundle.training_runs || {}).map(([key, run]) => `<div class="experiment-detail"><strong>${escapeHtml(run.model_name || key)}</strong><div class="config-strip">Batch size: ${escapeHtml(String(run.batch_size ?? '—'))} · LR: ${escapeHtml(String(run.lr ?? '—'))}<br>Epochs: ${escapeHtml(String(run.epochs ?? '—'))} · Max length: ${escapeHtml(String(run.max_len ?? '—'))}</div><p class="source-note">saved_models/${escapeHtml(key)}/eval_results.json · 构建快照</p></div>`).join('') || '<p class="muted">尚无训练配置记录。</p>';
}

async function refreshWorkbenchData() {
  const status = document.getElementById('dataSourceStatus');
  if (location.protocol === 'file:') { status.textContent = '数据来源：内置实验快照 · 支持完全离线浏览'; return; }
  const names = ['metrics', 'cases'];
  const results = await Promise.allSettled(names.map(async name => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch(`/api/${name}`, {signal: controller.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (!payload.success || !payload.data || typeof payload.data !== 'object') throw new Error('无效数据');
      if (name === 'metrics' && (!payload.data.models || !Array.isArray(payload.data.categories_comparison))) throw new Error('无效指标');
      if (name === 'cases' && !Array.isArray(payload.data.category_a_macbert_advantages)) throw new Error('无效案例');
      // Older backends may omit the optional generative case collection.
      bundle[name] = name === 'cases' ? {...bundle.cases, ...payload.data} : payload.data;
    } finally { clearTimeout(timer); }
  }));
  const refreshed = names.filter((_, index) => results[index].status === 'fulfilled');
  status.textContent = refreshed.length === 2 ? '抽取式指标与案例：已同步后端 · 生成式历史评测：原始实验文件构建快照' : refreshed.length ? `已同步${refreshed[0] === 'metrics' ? '抽取式指标' : '案例'}；其余使用内置快照` : '数据来源：内置实验快照 · 后端数据暂不可用';
  renderOverview(); syncBenchmarkSummary(); renderBenchmarkTable(); renderBenchmarkChart(); renderGalleryCases();
}

function initDualParadigm() {
  document.querySelectorAll('[data-navigate]').forEach(button => button.addEventListener('click', () => {
    switchTab(button.dataset.navigate);
    window.scrollTo({top: 0, behavior: 'smooth'});
  }));
  document.getElementById('experimentList').addEventListener('click', event => {
    const button = event.target.closest('[data-experiment]');
    if (!button) return;
    selectedExperiment = experiments[Number(button.dataset.experiment)];
    renderGeneration();
    document.querySelector(`[data-experiment="${button.dataset.experiment}"]`)?.focus();
  });
  document.getElementById('generationMetric').addEventListener('change', renderGenerationBars);
  document.getElementById('protocolCategory').addEventListener('change', renderProtocol);
  document.getElementById('exportExperiment').addEventListener('click', () => {
    if (!selectedExperiment) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(selectedExperiment, null, 2)], {type: 'application/json'}));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = `${selectedExperiment.id}.json`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.getElementById('viewGenerativeCases').addEventListener('click', () => {
    document.querySelector('.gallery-filter-btn[data-filter="generative"]').click();
    document.getElementById('caseSearch').value = '';
    renderGalleryCases(); switchTab('tab-gallery');
  });
  document.getElementById('caseSearch').addEventListener('input', renderGalleryCases);
  const nav = document.querySelector('.tab-nav-container');
  nav.setAttribute('role', 'tablist'); nav.setAttribute('aria-label', 'NER 工作台模块');
  nav.querySelectorAll('.tab-btn').forEach(button => {
    button.id = `nav-${button.dataset.tab}`; button.setAttribute('role', 'tab'); button.setAttribute('aria-controls', button.dataset.tab);
    const panel = document.getElementById(button.dataset.tab);
    panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', button.id);
  });
  nav.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const tabs = Array.from(nav.querySelectorAll('.tab-btn'));
    const current = tabs.indexOf(document.activeElement);
    if (current < 0) return;
    event.preventDefault();
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    tabs[index].click(); tabs[index].focus();
  });
  document.getElementById('themeToggleBtn').setAttribute('aria-label', '切换深色或浅色主题');
  renderOverview(); renderGeneration(); renderRetrievalAblation(); renderProtocol(); syncBenchmarkSummary();
  switchTab('tab-overview');
  refreshWorkbenchData();
}
