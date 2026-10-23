// Live Qwen + LoRA interaction. Historical experiment state stays independent.
const liveGenerationState = {running: false, available: false, result: null};

function validateLiveGeneration(data, inputText) {
  if (!data || data.source !== 'live' || data.text !== inputText || !Array.isArray(data.entities) || !Array.isArray(data.outputs)) {
    throw new Error('服务返回了无效的实时预测结果，请重试。');
  }
  const chars = Array.from(inputText);
  for (const entity of data.entities) {
    if (!Object.hasOwn(bundle.category_meta || {}, entity.category)
        || !Number.isInteger(entity.start) || !Number.isInteger(entity.end)
        || entity.start < 0 || entity.end < entity.start || entity.end >= chars.length
        || chars.slice(entity.start, entity.end + 1).join('') !== entity.text) {
      throw new Error('服务返回的实体位置无法与原文对齐，请重试。');
    }
  }
  for (const output of data.outputs) {
    if (!Object.hasOwn(bundle.category_meta || {}, output.category) || typeof output.output !== 'string'
        || !Array.isArray(output.entities) || !Array.isArray(output.warnings)
        || output.warnings.some(warning => typeof warning !== 'string') || typeof output.valid !== 'boolean') {
      throw new Error('服务返回的逐类别输出格式无效，请重试。');
    }
  }
  return data;
}

function renderLiveHighlight(text, entities) {
  const chars = Array.from(text);
  const cuts = [...new Set([0, chars.length, ...entities.flatMap(entity => [entity.start, entity.end + 1])])].sort((a, b) => a - b);
  return cuts.slice(0, -1).map((start, index) => {
    const end = cuts[index + 1];
    const value = escapeHtml(chars.slice(start, end).join(''));
    const covering = entities.filter(entity => entity.start <= start && entity.end >= end - 1);
    if (!covering.length) return value;
    const labels = [...new Set(covering.map(entity => bundle.category_meta[entity.category]?.cn || entity.category))].join(' / ');
    return `<span class="entity-span badge-${escapeHtml(covering[0].category)}" title="${escapeHtml(labels)} · [${start}, ${end - 1}]"><span class="ent-text">${value}</span><span class="ent-label">${escapeHtml(labels)}</span></span>`;
  }).join('');
}

function renderLiveGeneration(data) {
  document.getElementById('generativeHighlight').innerHTML = renderLiveHighlight(data.text, data.entities);
  document.getElementById('generativeEntityRows').innerHTML = data.entities.length ? data.entities.map(entity => `<tr><td>${escapeHtml(entity.text)}</td><td>${escapeHtml(bundle.category_meta[entity.category]?.cn || entity.category)}</td><td>${entity.start}</td><td>${entity.end}</td></tr>`).join('') : '<tr><td colspan="4">本次没有通过格式和原文对齐检查的实体，请结合下方原始输出查看。</td></tr>';
  document.getElementById('generativeRawOutputs').innerHTML = data.outputs.map(output => `<details class="live-output" ${output.valid ? '' : 'open'}><summary>${escapeHtml(bundle.category_meta[output.category]?.cn || output.category)} · ${output.valid ? `${output.entities.length} 个实体` : '输出校验未通过'}</summary><pre class="generation-raw">${escapeHtml(output.output)}</pre>${output.warnings.map(warning => `<p class="small">${escapeHtml(warning)}</p>`).join('')}</details>`).join('');
  const elapsed = typeof data.timing?.total_ms === 'number' ? `${(data.timing.total_ms / 1000).toFixed(2)} 秒` : '—';
  document.getElementById('generativeLiveSummary').textContent = `实时推理 · ${data.model} + ${data.adapter} · ${data.device} · ${data.entities.length} 个实体 · 总耗时 ${elapsed}${data.timing?.load_ms > 0 ? '（含首次加载）' : ''}`;
  const warnings = data.outputs.filter(output => !output.valid).length;
  const message = document.getElementById('generativeLiveMessage');
  message.textContent = warnings ? `识别完成，${warnings} 个类别的输出未通过校验，已保留原始文本供检查。可尝试缩短输入或增加生成长度。` : '识别完成 · 本次结果来自你本地训练后的 LoRA 模型。';
  message.dataset.error = 'false';
  document.getElementById('generativeLiveResults').hidden = false;
}

async function refreshGenerativeStatus() {
  const statusNode = document.getElementById('generativeModelStatus');
  const runButton = document.getElementById('runGenerative');
  liveGenerationState.available = false;
  runButton.disabled = true;
  if (location.protocol === 'file:') {
    statusNode.textContent = '当前为离线页面。运行 python app.py 后，通过 http://127.0.0.1:5000 使用生成式实时识别。';
    return;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch('/api/generative/status', {signal: controller.signal});
    if (response.status === 404) throw new Error('当前后端尚未提供生成式接口，请重启更新后的 python app.py。');
    if (!response.ok) throw new Error('无法连接生成式服务，请检查本地后端后刷新状态。');
    const payload = await response.json();
    if (!payload.success || typeof payload.data?.available !== 'boolean') throw new Error('模型状态响应无效，请重启本地服务。');
    const status = payload.data;
    liveGenerationState.available = status.available && !status.busy;
    statusNode.textContent = !status.available ? status.error : status.busy ? '模型正在加载或生成中，请稍后刷新状态。'
      : status.error ? `${status.error} 可重新尝试识别。`
      : status.loaded ? `${status.model} + ${status.adapter} 已就绪 · ${status.device} · 训练后权重`
      : `${status.model} + ${status.adapter} 权重已找到 · 首次识别时加载${status.device === 'auto' ? '，优先使用 GPU' : ` · ${status.device}`}`;
  } catch (error) {
    statusNode.textContent = error.name === 'AbortError' ? '模型状态检查超时，请稍后刷新。' : error.message;
  } finally {
    clearTimeout(timer);
    runButton.disabled = liveGenerationState.running || !liveGenerationState.available;
  }
}

async function runLiveGeneration() {
  if (liveGenerationState.running) return;
  const message = document.getElementById('generativeLiveMessage');
  const text = document.getElementById('generativeInput').value;
  const categories = Array.from(document.querySelectorAll('#generativeCategories input:checked')).map(input => input.value);
  message.dataset.error = 'false';
  liveGenerationState.result = null;
  document.getElementById('generativeLiveResults').hidden = true;
  if (!text.trim() || !categories.length || Array.from(text).length > 128 || /[\r\n\u0000]|@@|##/.test(text)) {
    message.textContent = '请输入 1–128 字的单行文本并选择至少一个类别；文本不能包含 @@ 或 ## 保留标记。';
    message.dataset.error = 'true';
    return;
  }
  liveGenerationState.running = true;
  const button = document.getElementById('runGenerative');
  button.disabled = true; button.textContent = '正在加载 / 生成，请稍候…';
  document.getElementById('refreshGenerativeStatus').disabled = true;
  document.getElementById('generativeInput').disabled = true;
  document.querySelectorAll('#generativeCategories input').forEach(input => { input.disabled = true; });
  ['selectAllGenerativeTypes', 'selectCommonGenerativeTypes', 'generativeTokenLimit'].forEach(id => { document.getElementById(id).disabled = true; });
  message.textContent = `正在使用训练后的 LoRA 模型识别 ${categories.length} 类实体，首次加载可能需要较长时间…`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180000);
  try {
    const response = await fetch('/api/generative/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, signal: controller.signal,
      body: JSON.stringify({text, categories, max_new_tokens: Number(document.getElementById('generativeTokenLimit').value)}),
    });
    if (response.status === 404) throw new Error('当前后端没有生成式预测接口，请重启更新后的 python app.py。');
    const payload = await response.json();
    if (!response.ok || !payload.success) throw new Error(payload.error || '生成式识别失败，请重试。');
    const data = validateLiveGeneration(payload.data, text);
    liveGenerationState.result = data;
    renderLiveGeneration(data);
  } catch (error) {
    message.dataset.error = 'true';
    message.textContent = error.name === 'AbortError'
      ? '等待超过 3 分钟。后端可能仍在处理，请稍后刷新模型状态；本次未显示预测结果。'
      : `识别未完成：${error.message}。本次未使用离线结果替代。`;
  } finally {
    clearTimeout(timer);
    liveGenerationState.running = false;
    button.textContent = '开始生成式识别 ↗';
    document.getElementById('refreshGenerativeStatus').disabled = false;
    document.getElementById('generativeInput').disabled = false;
    document.querySelectorAll('#generativeCategories input').forEach(input => { input.disabled = false; });
    ['selectAllGenerativeTypes', 'selectCommonGenerativeTypes', 'generativeTokenLimit'].forEach(id => { document.getElementById(id).disabled = false; });
    await refreshGenerativeStatus();
  }
}

function initLiveGeneration() {
  document.getElementById('generativeCategories').innerHTML = Object.entries(bundle.category_meta || {}).map(([category, meta]) => `<label><input type="checkbox" value="${escapeHtml(category)}" checked> ${escapeHtml(meta.cn || category)}</label>`).join('');
  const updateCount = () => { document.getElementById('generativeCharCount').textContent = `${Array.from(document.getElementById('generativeInput').value).length} / 128 字`; };
  document.getElementById('generativeInput').addEventListener('input', updateCount);
  updateCount();
  document.getElementById('selectAllGenerativeTypes').addEventListener('click', () => document.querySelectorAll('#generativeCategories input').forEach(input => { input.checked = true; }));
  document.getElementById('selectCommonGenerativeTypes').addEventListener('click', () => document.querySelectorAll('#generativeCategories input').forEach(input => { input.checked = ['name', 'company', 'address'].includes(input.value); }));
  document.getElementById('runGenerative').addEventListener('click', runLiveGeneration);
  document.getElementById('refreshGenerativeStatus').addEventListener('click', refreshGenerativeStatus);
  document.getElementById('exportGenerativePrediction').addEventListener('click', () => {
    if (!liveGenerationState.result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(liveGenerationState.result, null, 2)], {type: 'application/json'}));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'qwen-lora-prediction.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  refreshGenerativeStatus();
}
