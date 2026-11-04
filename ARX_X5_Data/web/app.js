const app = document.querySelector('#app');
let state, pending = false, networkError = '', guideOpen = localStorage.getItem('arx-guide') !== 'closed';
let checks = {switches:false, channels:[false,false,false,false]}, page = 'collect', preparedLoaded = false;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const phaseNames = {setup:'开工准备',ready:'等待采集',starting:'正在启动录制',recording:'正在录制',saving:'正在保存',reset:'场景复位',error:'需要处理',stopped:'遥操已结束'};
const boolText = (v, yes, no) => v === true ? yes : v === false ? no : '待核对';
const image = (path, alt) => path?.startsWith('/assets/') ? `<img class="reference" src="${esc(path)}" alt="${esc(alt)}">` : '';
async function action(name, extra = {}) {
  if(pending) return;
  if(name==='teleop_start' && !state.demo && state.config.teleop_start_notice && !window.confirm(state.config.teleop_start_notice+'\n\n确认现场已准备好，启动遥操？'))return;
  if(['record_discard','take_delete'].includes(name) && !window.confirm(name==='record_discard'?'删除本条录制？将结束并作废本条，遥操继续运行。':'删除这条采集记录？将移出有效计数并排除后续处理，原始文件保留，可从记录恢复。'))return;
  pending = true; networkError = '';
  if(['record_stop','record_discard'].includes(name))state.phase='saving';
  if(name==='record_start')state.phase='starting';
  render();
  try {
    const response = await fetch('/api/action', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:name,request_id:crypto.randomUUID(),...extra})});
    const data = await response.json();
    if(!response.ok) throw Error(data.error || '操作未完成');
    state = data;
  } catch(error) {networkError = `${error.message}。若连接中断，请先核对状态，勿重复提交设备操作。`;}
  finally {pending = false; await poll();}
}
async function poll(){
  try {
    const response = await fetch('/api/state');
    if(!response.ok) throw Error('本地服务连接失败');
    state = await response.json();
    if(!preparedLoaded){checks = {switches:state.switches,channels:[...state.channels]};preparedLoaded=true;}
    if(networkError.startsWith('连接已断开')) networkError='';
    render();
  } catch(error){networkError='连接已断开：请检查本地服务。界面保留的状态可能已过期，暂时不能操作。';render();}
}
function button(actionName, text, disabled=false, klass='primary', extra='') {
  return `<button class="${klass}" data-action="${actionName}" ${disabled || pending || state?.busy || networkError.startsWith('连接已断开') ? 'disabled':''} ${extra}>${text}</button>`;
}
const processingNames={queued:'等待后台处理',validating:'后台校验中',converting:'后台转换中',done:'处理完成',failed:'处理失败',validated:'校验通过',unknown:'状态待核对',demo:'演示数据',deleted:'已删除'};
function processingBadge(item){
  return `<span class="processing-badge ${esc(item.processing)}">${esc(processingNames[item.processing] || '状态待核对')}</span>${item.processing_error?`<small class="processing-error">${esc(item.processing_error)}</small>`:''}`;
}
function backgroundPanel(){
  const counts=state.processing_counts || {};
  const waiting=(counts.queued||0)+(counts.validating||0)+(counts.converting||0);
  const failures=counts.failed||0;
  return `<section class="background-panel panel" aria-live="polite"><div><h3>后台数据处理 <span class="tag">${waiting} 条处理中 · ${failures} 条失败</span></h3><p>本条写盘完成后即可复位并采集下一条，校验与转换自动排队处理。</p>${state.worker_alive===false?'<p class="processing-error">后台进程未运行，原始文件已保留；可继续采集，请联系负责人恢复后台处理。</p>':''}${failures?'<p class="processing-error">有条目处理失败，原始文件保留；请到采集记录查看原因。</p>':''}</div><button class="secondary" data-page="records">查看处理状态 →</button></section>`;
}
function setupPanel(){
  const c = state.config;
  return `<section class="panel setup-panel"><div class="section-head"><div><span class="eyebrow">BEFORE YOU START</span><h2>开工，按这三步来</h2></div><span class="tag">每日首次使用</span></div>
  <div class="setup-grid"><article class="instruction"><span class="step-number">01</span><h3>打开设备开关</h3><p>${esc(c.switch_instructions)}</p>${image(c.switch_image,'设备开关位置与顺序')}<label class="check"><input type="checkbox" data-check="switches" ${checks.switches?'checked':''}>已按现场顺序打开开关</label></article>
  <article class="instruction"><span class="step-number">02</span><h3>启动四路 CAN</h3><p>在 CAN Doctor 点击 <b>检测设备 / Detect Devices</b>，核对四路后执行 <b>一键启动 / Start All</b>。</p><button class="secondary" data-can-open>连接 CAN Doctor ↗</button><small>以下为人工确认，不是实时检测。</small>${c.channels.map((ch,i)=>`<div class="channel"><label class="check"><input type="checkbox" data-channel="${i}" ${checks.channels[i]?'checked':''}><span><b>${esc(ch.name)}</b> · ${esc(ch.device)}<small>正常标志：${esc(ch.success)}</small></span></label>${image(ch.image,ch.name+' 正常状态')}</div>`).join('')}</article>
  <article class="instruction"><span class="step-number">03</span><h3>连接遥操，做一条试采</h3><p>确认前两步后，核对程序状态，再启动遥操。试采结束后检查本地文件。</p>${button('prepare','确认开工检查',!checks.switches || !checks.channels.every(Boolean),'secondary')}${button('status','核对设备状态',!state.teleop_configured,'secondary')}${button('teleop_start',state.teleop_running===true?'遥操已运行':'启动遥操',!state.teleop_configured || !state.switches || state.teleop_running!==false || state.recording!==false || !['setup','stopped'].includes(state.phase))}${state.config.teleop_start_notice?`<p class="teleop-notice">${esc(state.config.teleop_start_notice)}</p>`:''}<small>收工：结束并保存 → 手动回正双臂并稳定 → 点击顶部“结束遥操”。未退出时按提示处理，再次点击。</small></article></div><div class="shutdown-note"><b>收工步骤</b><p>${esc(c.shutdown_instructions)}</p></div></section>`;
}
function startBlockReason(s){
  if(!s.configured)return '录制接口尚未接入，请联系现场负责人。';
  if(!s.switches || !s.channels.every(Boolean))return '请先在下方完成开关与四路 CAN 的开工确认。';
  if(s.teleop_running!==true)return s.teleop_detail || '正在等待真实遥操反馈，请核对状态。';
  if(s.recording!==false)return '录制状态尚未确认，请核对现有程序。';
  if(s.busy || pending)return '正在处理上一项操作，请稍候。';
  return '';
}
function currentCard(){
  const s=state; let title, subtitle, control;
  switch(s.phase){
    case 'setup': title='准备好，再开始';subtitle='完成开工检查并核对设备状态，然后启动遥操。';control=button('status','核对设备状态',!s.teleop_configured);break;
    case 'ready':title=s.trial_verified?'下一条，准备就绪':'先做一条试采';subtitle=s.trial_verified?'摆好场景后，点击开始。只录制正式任务过程。':'按正式任务操作一次，结束后检查保存目录中的数据文件。';control=button('record_start',s.trial_verified?'开始本条录制':'开始试采',!s.configured || !s.switches || !s.channels.every(Boolean) || s.teleop_running!==true || s.recording!==false);break;
    case 'stopped':title='遥操已结束';subtitle='数据已保留。再次采集前请启动遥操；启动程序仍包含双臂回零。';control=button('teleop_start','重新启动遥操',!s.teleop_configured || !s.switches || !s.channels.every(Boolean) || s.recording!==false);break;
    case 'starting':title='正在启动录制';subtitle='等待录制程序确认，请稍候。';control='<div class="waiting"><span class="spinner"></span>正在确认设备反馈</div>';break;
    case 'recording':title=s.trial?'试采正在录制':'正在录制任务';subtitle='完成任务后点击结束并保存；场景复位放在保存完成之后。';control=`<div class="capture-actions">${button('record_stop','结束并保存',false,'primary stop')}${button('record_discard','删除本条',!s.discard_configured,'secondary delete-take')}</div>`;break;
    case 'saving':title='正在完成本条写盘';subtitle='文件关闭并写盘完成后即可复位，校验与转换将在后台继续。';control='<div class="waiting"><span class="spinner"></span>等待文件写盘完成</div>';break;
    case 'reset':title='用机械臂，恢复场景';subtitle='遥操保持开启，当前不录制，请用机械臂恢复场景。';control=button('reset','场景已复位',s.recording!==false || s.teleop_running!==true);break;
    default:title='先处理当前问题';subtitle='操作结果尚未确认。保留已有文件，查看提示后核对现有程序。';control=button('status','重新核对状态',!s.teleop_configured,'secondary');
  }
  return `<section class="capture-card ${esc(s.phase)}"><div class="capture-top"><span class="eyebrow">${s.trial_verified?'COLLECTION SESSION':'FIRST TAKE CHECK'}</span><span class="status-pill ${s.phase==='recording'?'live':''}"><i></i>${esc(phaseNames[s.phase])}</span></div><div class="capture-body"><div class="orbit"><span>${s.phase==='recording'?'●':s.phase==='reset'?'↺':s.phase==='error'?'!':'▶'}</span></div><h1>${title}</h1><p>${subtitle}</p>${s.phase==='recording'?'<div class="timer" id="timer">00:00</div>':''}${control}${s.phase==='ready' && startBlockReason(s)?`<p class="blocked-reason" role="status">${esc(startBlockReason(s))}</p>`:''}<div class="card-foot">${s.phase==='reset'?'复位完成后仍需手动点击，才会开始下一条。':'数据保存于本机 · 录制与遥操分别控制'}</div></div></section>`;
}
function render(){
  if(!state){app.innerHTML=`<p class="loading">${esc(networkError || '正在连接本地工作台…')}</p>`;return;}
  const s=state, latest=s.saved.at(-1), needSetup=!s.switches || !s.channels.every(Boolean) || s.phase==='setup';
  app.innerHTML=`<div class="shell"><aside class="sidebar"><a class="brand" href="/"><span class="brand-symbol">A<span>↗</span></span><div>ARX<small>DATA COLLECTION</small></div></a><div class="side-label">工作空间</div><button class="nav-item ${page==='collect'?'active':''}" data-page="collect"><span>◫</span> 数采工作台 <span class="nav-dot"></span></button><button class="nav-item ${page==='records'?'active':''}" data-page="records"><span>▤</span> 本地采集记录 <em>${s.saved.length}</em></button><a class="nav-item" href="/browse"><span>▷</span> 数据浏览与审核</a><a class="nav-item" href="/guide" target="_blank"><span>↗</span> 一页速查教程</a><div class="side-bottom"><div class="local-mark"><i></i> 本机工作空间</div><p>遥操持续运行<br>每条任务独立保存</p><span class="version">ARX WORKBENCH / 01</span></div></aside>
  <main><header><div><div class="breadcrumb">工作空间 <span>/</span> ${page==='collect'?'数采工作台':'本地采集记录'}</div><h2>${page==='collect'?'让每一条采集，都有序完成。':'本次保存，清楚可查。'}</h2></div><div class="header-tools"><span class="mode ${s.demo?'demo':''}">${s.demo?'演示模式 · 无设备数据':'设备模式 · 本地运行'}</span>${button('teleop_stop','结束遥操',!s.teleop_stop_configured || s.recording!==false || ['recording','saving','starting'].includes(s.phase) || !(s.teleop_running || s.teleop_process_running),'secondary end-teleop','title="双臂手动回正并稳定后正常退出；录制中请先结束并保存"')}<button class="icon-button" data-action="status" ${pending || s.busy || !s.teleop_configured?'disabled':''} title="读取设备真实状态">↻ 核对状态</button></div></header>
  ${!s.configured?'<div class="notice warning"><b>录制控制尚未接入</b><span>CAN Doctor 和遥操可分别操作；本页的正式录制按钮需完成真实录制接入后使用。</span></div>':''}
  ${s.demo?'<div class="notice demo-note"><b>这是操作演示</b><span>只生成标记为 demo 的示例文件；页面状态不代表真实设备。</span></div>':''}
  ${s.notice?`<div class="notice info" role="status">${esc(s.notice)}</div>`:''}
  ${networkError || s.error?`<div class="notice error" role="alert"><b>需要处理</b><span>${esc(networkError || s.error)}</span></div>`:''}
  <section class="metrics" aria-live="polite"><div class="metric"><span>遥操状态</span><strong><i class="dot ${s.teleop_running===true?'green':''}"></i>${boolText(s.teleop_running,'运行中','未启动')}</strong><small>${esc(s.demo?'演示程序状态':s.teleop_detail || '来自真实进程与遥操反馈')}</small></div><div class="metric"><span>录制状态</span><strong><i class="dot ${s.recording===true?'red':''}"></i>${s.phase==='saving'?'写盘中':boolText(s.recording,'录制中','未录制')}</strong><small>${esc(phaseNames[s.phase])}</small></div><div class="metric path"><span>本地保存目录</span><strong title="${esc(s.config.save_dir)}">${esc(s.config.save_dir || '尚未配置')}</strong><small>沿用现有采集程序的保存规则</small></div><div class="metric count"><span>本地已存正式条数</span><strong>${String(s.saved_count).padStart(2,'0')}<small> 条</small></strong><small>校验通过 ${s.validated_count || 0} 条 · 试采单独标记</small></div></section>${backgroundPanel()}
  ${page==='collect'?`<section class="can-entry panel"><div><span class="eyebrow">STEP 01 / DEVICE CONNECTION</span><h3>先连接四路 CAN，再启动遥操</h3><p>点击右侧按钮，在工作台内完成 Detect Devices 和 Start All，然后按开工引导逐路确认。</p></div><button class="primary" data-can-open>连接 CAN Doctor ↗</button></section><div class="workflow"><span class="${needSetup?'current':''}"><b>1</b> 开工检查</span><i></i><span class="${['ready','recording','starting'].includes(s.phase)?'current':''}"><b>2</b> 任务采集</span><i></i><span class="${s.phase==='saving'?'current':''}"><b>3</b> 本地保存</span><i></i><span class="${s.phase==='reset'?'current':''}"><b>4</b> 遥操复位</span><span class="loop-note">↺ 循环采集</span></div>
  <div class="workspace">${currentCard()}<aside class="context"><section class="panel next-panel"><span class="eyebrow">OPERATOR NOTES</span><h3>当前怎么做</h3><ol><li><b>只录任务过程</b><p>开始录制后完成任务，结束时先保存。</p></li><li><b>保存后再复位</b><p>遥操保持开启，用机械臂恢复场景。</p></li><li><b>确认后录下一条</b><p>复位动作不进入正式任务数据。</p></li></ol></section><section class="panel latest"><div class="section-head"><h3>最近保存</h3><span class="tag">本地</span></div>${latest?`<span class="saved-badge">${latest.deleted?'本条已删除，原始文件保留':'✓ 已保存到本地'}</span>${processingBadge(latest)}<h4 title="${esc(latest.id)}">${esc(String(latest.id).split('/').filter(Boolean).slice(-2).join(' / '))}</h4><p class="file-path">${esc(latest.path)}</p><small>${latest.trial?'试采数据':'正式数据'} · ${new Date(latest.time*1000).toLocaleTimeString('zh-CN')}</small>${takeAction(latest)}`:'<div class="empty-icon">▱</div><p>还没有保存记录</p><small>完成第一条后，文件位置会显示在这里。</small>'}</section></aside></div>
  ${!s.trial_verified && latest?.trial && !latest.deleted?`<section class="notice trial"><div><b>试采已保存，请先检查文件</b><p>在设备电脑核对文件可读、内容和时长符合要求，再确认进入正式采集。</p></div>${button('verify_trial','文件已核对，进入正式采集',false,'secondary')}</section>`:''}
  <section class="guide-toggle"><div><h3>开工操作引导</h3><p>开关 → CAN Doctor → 遥操 → 试采核对</p></div><button class="text-button" id="toggle-guide" aria-expanded="${guideOpen || needSetup}">${guideOpen || needSetup?'收起教程 ↑':'展开教程 ↓'}</button></section>${guideOpen || needSetup?setupPanel():''}`:records()}
  <footer><span>ARX · 数采工作台</span><span>结束一条 ≠ 关闭遥操 <i>·</i> 数据先存本地</span></footer></main></div>`;
  bind();tick();
}
function takeAction(item){
  const episode=String(item.path).split('/').filter(Boolean).slice(-2).join('/');
  const browseLink=!state.demo&&/episode_\d+$/.test(episode)?`<a class="secondary browse-take" href="/browse#${encodeURIComponent(episode)}">浏览与审核</a> `:'';
  if(item.deleted)return browseLink+(item.can_restore?button('take_restore','恢复本条',false,'secondary restore-take',`data-take-id="${esc(item.id)}"`):'<small>录制中已作废</small>');
  return browseLink+button('take_delete','删除本条',false,'secondary delete-take',`data-take-id="${esc(item.id)}"`);
}
function records(){return `<section class="panel records"><div class="section-head"><h3>本地保存记录</h3><span class="tag">${state.saved.length} 条（含试采）</span></div><p>本地写盘完成即列入记录；后台处理状态单独更新，处理失败的条目保留原始文件。</p><div class="table-wrap"><table><thead><tr><th>条目编号</th><th>类型</th><th>保存时间</th><th>后台处理</th><th>本地路径</th><th>操作</th></tr></thead><tbody>${[...state.saved].reverse().map(x=>`<tr class="${x.deleted?'deleted-row':''}"><td title="${esc(x.id)}">${esc(String(x.id).split('/').filter(Boolean).slice(-2).join(' / '))}</td><td><span class="tag">${x.trial?'试采':'正式'}</span></td><td>${new Date(x.time*1000).toLocaleString('zh-CN')}</td><td>${processingBadge(x)}</td><td class="file-path">${esc(x.path)}</td><td>${takeAction(x)}</td></tr>`).join('') || '<tr><td colspan="6" class="empty">尚无保存记录</td></tr>'}</tbody></table></div></section>`;}
function bind(){
  app.querySelectorAll('[data-can-open]').forEach(el=>el.onclick=openCanDoctor);
  app.querySelectorAll('[data-action]').forEach(el=>el.onclick=()=>action(el.dataset.action,el.dataset.action==='prepare'?checks:el.dataset.takeId?{id:el.dataset.takeId}:{}));
  app.querySelectorAll('[data-page]').forEach(el=>el.onclick=()=>{page=el.dataset.page;render();});
  app.querySelectorAll('[data-check]').forEach(el=>el.onchange=()=>{checks.switches=el.checked;render();});
  app.querySelectorAll('[data-channel]').forEach(el=>el.onchange=()=>{checks.channels[Number(el.dataset.channel)]=el.checked;render();});
  const toggle=app.querySelector('#toggle-guide');if(toggle)toggle.onclick=()=>{guideOpen=!guideOpen;localStorage.setItem('arx-guide',guideOpen?'open':'closed');render();};
}
function tick(){const el=document.querySelector('#timer');if(el && state?.started_at){const seconds=Math.max(0,Math.floor(Date.now()/1000-state.started_at));el.textContent=`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;}}
poll();setInterval(poll,2500);setInterval(tick,1000);
// Poll the read-only device adapter separately; never replay commands on refresh.
setInterval(()=>{if(state?.teleop_configured && !pending && !state.busy && state.phase!=='error')action('status');},10000);

// Keep this iframe outside the polled app tree: do not reload a hardware connection every 2.5s.
let canOpening=false;
async function openCanDoctor(){
  if(canOpening)return;
  const dialog=document.querySelector('#can-dialog');
  const frame=document.querySelector('#can-frame');
  const target=state.can_doctor_embed_url;
  document.querySelector('#can-mode').textContent=state.can_doctor_demo?'模拟 CAN 页面 · 无设备连接':'真实 CAN Doctor · 操作影响实际设备';
  document.querySelector('#can-channel-reference').innerHTML=state.config.channels.map(ch=>`<div class="can-reference"><b>${esc(ch.name)} · ${esc(ch.device)}</b><p>正常标志：${esc(ch.success)}</p></div>`).join('');
  const external=document.querySelector('#can-external');
  external.hidden=!target;
  if(target)external.href=target;
  document.querySelector('#can-unconfigured').hidden=Boolean(target);
  frame.hidden=!target;
  if(!dialog.open)dialog.showModal();
  if(!target)return;
  canOpening=true;
  const message=document.querySelector('#can-unconfigured');
  message.hidden=false;message.textContent='正在连接 CAN Doctor 网页服务…';
  try{
    const response=await fetch('/api/can/open',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const data=await response.json();
    if(!response.ok)throw Error(data.error || '网页服务未就绪');
    if(frame.getAttribute('src')!==data.url)frame.src=data.url;
    frame.hidden=false;message.hidden=true;
  }catch(error){message.textContent=error.message;message.hidden=false;}
  finally{canOpening=false;}
}
document.querySelector('#can-close').onclick=()=>document.querySelector('#can-dialog').close();
