const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const app = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
function context() {
  const ctx = { escapeHtml: value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c])) };
  vm.createContext(ctx);
  vm.runInContext(app.slice(app.indexOf('function qaSourceMeta('), app.indexOf('function renderQaResponseMeta(')), ctx);
  vm.runInContext(app.slice(app.indexOf('function renderQaSources('), app.indexOf('function setQaProgress(')), ctx);
  return ctx;
}
function message() {
  return { role: 'assistant', content: 'Legacy text', answer_data: {paragraphs: [
    {kind: 'answer', text: 'Add demo links.', source_refs: [13]},
  ]}, sources: Array.from({length: 13}, (_, i) => ({chat_type: i === 12 ? 'private' : 'group',
    chat_title: i === 12 ? 'Lin' : 'Internships', sender: 'Lin', time: '10:00', text: 'Original message'})) };
}

test('search trace distinguishes hits from automatically expanded context', () => {
  const html = context().renderQaSearchTrace({query_plan: {tool_first: true}, trace: [
    {tool: 'search', returned: 7, included: 7, anchor_count: 1, context_count: 6},
  ]});
  assert.match(html, /命中 1 条 · 自动补充前后文 6 条/);
});

test('task requirements are inspectable and escaped even without a trace', () => {
  const html = context().renderQaSearchTrace({current_task: {request: '<排除联系人>',
    expected_result: '比较其余联系人', requirements: [{text: '<要求>', source_turn: 2,
      source_quote: '<script>原话</script>'}]}});
  assert.match(html, /本轮理解 · &lt;排除联系人&gt;/);
  assert.match(html, /依据用户第 2 次发言/);
  assert.doesNotMatch(html, /<script>/);
});

test('trace distinguishes original messages from scored windows', () => {
  const html = context().renderQaSearchTrace({trace: [{tool: 'search',
    retrieval_counts: {message_candidate_count: 2400, window_candidate_count: 120}}]});
  assert.match(html, /候选消息 2400 条 → 重排窗口 120 个/);
});

test('answer checks display escaped issues without presenting a correctness guarantee', () => {
  const html = context().renderQaSearchTrace({query_plan: {query: 'SG?'}, trace: [],
    quality_status: 'unresolved', answer_reviews: [{attempt: 1, decision: 'revise',
      elapsed_ms: 42, issues: [{reason: '<script>bad</script>'}]}]});
  assert.match(html, /模型复核 1/);
  assert.match(html, /现有证据需重新作答/);
  assert.match(html, /检查未通过，未发布草稿结论/);
  assert.match(html, /&lt;script&gt;/);
  assert.doesNotMatch(html, /<script>|保证正确/);
});

test('citation 13 uses private source metadata and can expand the original message', () => {
  const html = context().renderQaAnswer(message());
  assert.match(html, /\[13\] · 私聊 · Lin · 10:00 · Lin/);
  assert.match(html, /<details class="qa-citation">/);
  assert.match(html, /Original message/);
  assert.doesNotMatch(html, /群聊|Internships/);
});

test('source metadata comes from retrieved data, not extra model fields', () => {
  const msg = message();
  msg.answer_data.paragraphs[0].chat_title = 'Wrong group';
  const html = context().renderQaAnswer(msg);
  assert.doesNotMatch(html, /Wrong group/);
  assert.match(html, /私聊 · Lin/);
});

test('legacy and invalid structured answers fall back to escaped plain text', () => {
  const ctx = context();
  const msg = message();
  msg.answer_data.paragraphs[0].source_refs = [14];
  assert.match(ctx.renderQaAnswer(msg), /Legacy text/);
  assert.doesNotMatch(ctx.renderQaAnswer(msg), /qa-citation/);
  assert.match(ctx.renderQaAnswer({role: 'assistant', content: '<img>'}), /&lt;img&gt;/);
});

test('answer, source title and original message cannot inject HTML', () => {
  const msg = message();
  msg.answer_data.paragraphs[0].text = '<script>alert(1)</script>';
  msg.sources[12].chat_title = '<img src=x onerror=alert(1)>';
  msg.sources[12].text = '<iframe>';
  const html = context().renderQaAnswer(msg);
  assert.doesNotMatch(html, /<script|<img|<iframe/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /&lt;iframe&gt;/);
});

test('source list numbers match paragraph references without truncation', () => {
  const html = context().renderQaSources(message());
  assert.match(html, /\[1\] · 群聊/);
  assert.match(html, /\[13\] · 私聊/);
  assert.equal((html.match(/class="qa-source"/g) || []).length, 13);
});

test('query trace remains visible with no sources and escapes model strings', () => {
  const html = context().renderQaSources({retrieval: {query_plan: {query: '<img>', people: [], clarification: '请确认人物'},
    trace: [{tool: 'timeline', returned: 0, included: 0, matched: 0, elapsed_ms: 1, complete: true}]}});
  assert.match(html, /查询过程/);
  assert.match(html, /请确认人物/);
  assert.match(html, /消息时间线/);
  assert.doesNotMatch(html, /<img>/);
});

test('missing evidence renders without invented citations', () => {
  const msg = message();
  msg.answer_data.paragraphs = [{kind: 'limitation', text: 'No deadline found.', source_refs: []}];
  assert.doesNotMatch(context().renderQaAnswer(msg), /qa-citation/);
});

test('per-call scope and discovered people are visible and escaped', () => {
  const html = context().renderQaSearchTrace({query_plan: {tool_first: true, people: []}, trace: [
    {tool: 'find_people', returned: 2, unit: '个人物', complete: true},
    {tool: 'search', parameters: {query: '周一', filters: {chat_type: 'group', chat_keyword: '<img>',
      since: '2026-10-01T00:00:00+08:00', until: null}}},
  ]});
  assert.match(html, /查找人物/);
  assert.match(html, /2 个人物/);
  assert.match(html, /2026-10-01/);
  assert.match(html, /&lt;img&gt;/);
  assert.doesNotMatch(html, /<img>/);
});

test('trace distinguishes candidate windows from messages and escapes degradation notes', () => {
  const html = context().renderQaSearchTrace({query_plan: {people: []}, scope: '群聊 · NTU', trace: [
    {tool: 'shared_windows', returned: 18, included: 18, matched: 19, complete: false,
      note: '候选不是互动次数 <img>'},
    {tool: 'search', returned: 2, note: '重排未完成，使用已有排序'},
  ]});
  assert.match(html, /19 个候选时间窗/);
  assert.match(html, /群聊 · NTU/);
  assert.match(html, /重排未完成/);
  assert.match(html, /&lt;img&gt;/);
  assert.doesNotMatch(html, /<img>/);
});

test('done event retains structured content before any history reload', () => {
  const ctx = context();
  Object.assign(ctx, {state: {}, renderQaMessages() {}, setQaProgress() {}});
  vm.runInContext(app.slice(app.indexOf('function handleQaStreamEvent('), app.indexOf('function renderQaMessages(')), ctx);
  const result = {};
  const data = message();
  ctx.handleQaStreamEvent({event: 'done', processing_ms: 10, answer: data.content,
    answer_data: data.answer_data, sources: data.sources}, result);
  assert.deepEqual(result.answer_data, data.answer_data);
  assert.match(ctx.renderQaAnswer({...result, role: 'assistant'}), /\[13\] · 私聊/);
});

test('pending retrieval can expand before the first completed tool or query plan', () => {
  const html = context().renderQaSources({id: 'run-1', pending: true, progress: '模型决定第 1 步检索'});
  assert.match(html, /查询过程 · 0 步工具调用 · 进行中/);
  assert.match(html, /data-qa-detail="trace:run-1"/);
  assert.match(html, /模型决定第 1 步检索/);
  assert.doesNotMatch(html, /对象：全库|未限定日期/);
});

test('step details include escaped parameters, result previews, coverage and timing', () => {
  const html = context().renderQaSearchTrace({trace: [{tool: 'query_messages', returned: 60,
    included: 30, matched: 60, next_offset: 30, complete: false, elapsed_ms: 1234,
    parameters: {sql: 'SELECT * WHERE text < ? ', params: ['<img>']},
    preview_total: 60, preview: [{chat_title: '<script>', sender: 'Lin', time: '12:00', text: '<iframe>'}]}],
    active_step: {phase: 'model', message: '模型决定第 2 步检索', started_at_ms: Date.now()}}, {id: 'run-2', pending: true});
  assert.match(html, /qa-trace-step/);
  assert.match(html, /只读 SQL/);
  assert.match(html, /1.2s/);
  assert.match(html, /返回 60 条 · 纳入 30 条证据 · 匹配 60 条 · 部分覆盖/);
  assert.match(html, /后续分页起点：30/);
  assert.match(html, /结果预览 · 1\/60/);
  assert.match(html, /等待模型选择工具/);
  assert.match(html, /data-qa-stage-start/);
  assert.match(html, /&lt;script&gt;/);
  assert.doesNotMatch(html, /<script>|<img>|<iframe>/);
});

test('progress updates diagnostics without rerendering the conversation or stopping the job', () => {
  const ctx = context();
  const calls = [];
  Object.assign(ctx, {
    updateQaTraceNode(msg) { calls.push(msg.retrieval); },
    updateQaProgressFromMessage() {}, updateQaElapsedNode() {},
    renderQaMessages() { throw Error('must not rerender conversation'); },
  });
  vm.runInContext(app.slice(app.indexOf('function handleQaStreamEvent('), app.indexOf('function renderQaMessages(')), ctx);
  const msg = {id: 'run-3', pending: true};
  const retrieval = {trace: [{tool: 'search'}], active_step: {phase: 'model'}};
  ctx.handleQaStreamEvent({event: 'progress', message: '下一步', retrieval}, msg);
  assert.equal(msg.retrieval, retrieval);
  assert.equal(msg.pending, true);
  assert.deepEqual(calls, [retrieval]);
  ctx.handleQaStreamEvent({event: 'progress', message: '仍在执行'}, msg);
  assert.equal(msg.retrieval, retrieval);
});

test('trace refresh preserves expanded steps and scroll position', () => {
  const ctx = context();
  const oldNodes = [{dataset: {qaDetail: 'trace:run-4'}, open: true},
    {dataset: {qaDetail: 'trace:run-4:call-1'}, open: true}];
  const newNodes = [...oldNodes.map(n => ({dataset: n.dataset, open: false})),
    {dataset: {qaDetail: 'trace:run-4:call-2'}, open: false}];
  let replaced = false;
  const body = {scrollTop: 137};
  const root = {
    querySelectorAll(selector) { return replaced ? newNodes : oldNodes; },
    querySelector() { return body; },
    set innerHTML(value) { replaced = true; body.scrollTop = 0; },
  };
  ctx.qaMessages = {querySelector() { return root; }};
  ctx.CSS = {escape: value => value};
  ctx.updateQaTraceNode({id: 'run-4', pending: true, retrieval: {trace: []}});
  assert.deepEqual(newNodes.map(n => n.open), [true, true, false]);
  assert.equal(body.scrollTop, 137);
});

test('trace exposes stage timings, batch wall times and model input size separately', () => {
  const html = context().renderQaSearchTrace({trace:[{tool:'search',timings:{
    embedding_ms:200,vector_match_ms:50,rerank_ms:5000,context_expansion_ms:25},
    rerank_batch_timings:[{batch:1,candidates:48,elapsed_ms:4200,attempts:1}]}],
    model_metrics:[{phase:'decision',elapsed_ms:1000,prompt_tokens:2000,request_chars:4000}],
    answer_timings:[{elapsed_ms:2000,source_count:12}]});
  assert.match(html,/本地向量匹配/);
  assert.match(html,/重排（墙钟时间）/);
  assert.match(html,/重排批次耗时 · 1 批/);
  assert.match(html,/第 1 批 · 48 条 · 4.2s/);
  assert.match(html,/模型请求耗时与输入量/);
  assert.match(html,/生成回答 1 · 2.0s · 12 条证据/);
});

test('stream diagnostics distinguish first event, first content and completion', () => {
  const html = context().renderQaSearchTrace({query_plan:{query:'test'},transport_calls:[{
    phase:'answer',first_event_ms:100,first_content_ms:3500,total_ms:8000,request_chars:1200,prompt_tokens:600,
  }]});
  assert.match(html,/首个事件 0.1s/);
  assert.match(html,/首段内容 3.5s/);
  assert.match(html,/完整响应 8.0s/);
  assert.match(html,/含工具定义/);
});

test('multi-query trace distinguishes title filters and per-query recall', () => {
  const html = context().renderQaSearchTrace({trace:[{tool:'search',parameters:{
    query:'Application help',queries:['portfolio','reference'],filters:{chat_title_contains:'NTU'}},
    query_results:[{query:'portfolio',semantic_count:3,fts_count:2,like_count:1},{query:'<reference>',semantic_count:4,fts_count:1,like_count:0}]
  }]});
  assert.match(html,/内容查询列表/);
  assert.match(html,/名称包含：NTU/);
  assert.match(html,/各查询召回结果 · 2 条查询/);
  assert.match(html,/&lt;reference&gt;/);
});
