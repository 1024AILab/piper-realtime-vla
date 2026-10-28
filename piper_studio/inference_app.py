"""Piper Studio inference console; shared devices/code, separate capture entry point."""
from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr
import numpy as np

from .ui_components import esc, plot, table
from .config import Config
from .inference import InferenceEngine, PROMPT
from .preview_image import PreviewImage

PHASES = {'idle': '待命', 'preview': '仅预览预测', 'preparing': '正在准备从臂',
          'executing': '正在执行', 'returning': '复位到起始位', 'paused': '已暂停', 'fault': '已停止 · 请检查'}


def status_html(s):
    sample = s['sample']
    def duration(v):
        return f'{v:.0f}<small>ms</small>' if v is not None else '—'
    cards = [('运行状态', PHASES[s['phase']]), ('推理往返', duration(s['latency_ms'])),
             ('模型预测', f'{s["plans"]}<small>次</small>'),
             ('下发动作', f'{s["commands"]}<small>条 / {s["command_hz"]:.1f} Hz</small>')]
    stats = ''.join(f'<div class="metric"><span>{k}</span><strong>{v}</strong></div>' for k, v in cards)
    fresh = sample and sample.follower is not None and sample.feedback_age < 0.5 and not sample.error
    pills = [('从臂反馈', bool(fresh)), ('场景相机', s['camera_ages'].get('image', 999) < 0.6),
             ('腕部相机', s['camera_ages'].get('wrist_image', 999) < 0.6), ('模型服务', bool(s['model']))]
    badges = ''.join(f'<span class="pill {"good" if good else "muted"}">{"●" if good else "○"} {label}</span>' for label, good in pills)
    prediction_error = s['prediction_note'] if s['prediction_note'].startswith('禁止执行') else ''
    issues = list(dict.fromkeys(filter(None, [*s['issues'], s['error'], prediction_error])))
    error = '<div class="error-banner">' + '<br>'.join(esc(x) for x in issues) + '</div>' if issues else ''
    if not s['connected']:
        error = ''
    note = ('<div class="capture-hint">' + esc(s['prediction_note']) + '</div>'
            if s['prediction_note'] and not prediction_error else '')
    ended = ('<div class="capture-hint">' + esc(s['last_stop_reason']) + '</div>'
             if s['last_stop_reason'] and s['phase'] not in {'executing', 'preview', 'returning'} else '')
    return f'<div class="session-bar">{badges}</div><div class="metrics">{stats}</div>{note}{error}{ended}'


def build_app(engine):
    control_cache = {}
    def checked(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def refresh(request: gr.Request):
        s = engine.snapshot(request.session_hash)
        sample = s['sample']
        q = sample.follower if sample else None
        pred = s['prediction']
        active = s['phase'] in {'executing', 'preview', 'preparing', 'returning'} or s['inflight']
        ready = not s['issues'] and bool(s['model'])
        enabled = bool(sample and sample.enabled)
        helptext = ('从臂已使能 · 执行已准备' if s['prepared'] else
                    '从臂已使能 · 尚未准备平台执行' if enabled else '从臂未使能 · 可先预览预测')
        if s['phase'] in {'preview', 'executing'}:
            helptext += f'　·　已运行 {s["elapsed"]:.0f} 秒'
            helptext += (f'　·　剩余 {s["remaining"]:.0f} 秒' if s['remaining'] is not None else '　·　完成后手动停止')
        if s['inflight']:
            helptext += ('　·　后台更新预测，控制继续' if s['phase'] == 'executing' and s['buffer_remaining_s'] > 0
                         else '　·　等待本次模型结果')
        details = f'本次记录：`{s["run"]}`' if s['run'] else '每次运行自动保存输入画面、预测和实际命令，可在“运行回看”查看。'
        times = ('模型计算 ' + f'{s["server_ms"]:.0f} ms · ' if s['server_ms'] is not None else '')
        times += f'连续控制目标 {s["control_rate_hz"]} Hz · 重新预测间隔 {engine.chunk_steps / engine.config.fps:.1f} 秒。'
        times += f'夹爪速度上限 {engine.config.max_gripper_speed_mm_s:g} mm/s · 行程 0–{engine.config.gripper_max_um / 1000:g} mm。'
        if s['phase'] == 'executing':
            times += f'有效轨迹余量 {s["buffer_remaining_s"]:.2f} 秒（从结果就绪计时） · 控制错过周期 {s["missed_control_ticks"]} 次。'
            if s['inference_wait_s'] is not None:
                times += f'新预测已等待 {s["inference_wait_s"]:.2f} / 1.50 秒。'
        times += '前 6 维为关节角度 °，夹爪为 mm；图中显示原始预测，实际发送另受速度限制。'
        controls = (ready and not active, ready and not active and s['prepared'] and enabled,
                    s['connected'] and not active, not active, not s['connected'],
                    s['connected'] and not active, s['connected'] and enabled and s['home'] is not None and not active)
        previous = control_cache.get(request.session_hash)
        # Do not resend button properties every 250 ms. Keep operator-owned
        # controls stable while live images/telemetry refresh around them.
        updates = [gr.update(interactive=value) if previous is None or previous[i] != value else gr.skip()
                   for i, value in enumerate(controls)]
        control_cache[request.session_hash] = controls
        home_label = ('起始位：' + s['home_source']) if s['home'] is not None else '首次开始执行时会自动记住起始位，也可以手动记录。'
        return (s['images'].get('image'), s['images'].get('wrist_image'), status_html(s),
                table(q, s['last_sent'], ('从臂实际', '实际下发')), plot(pred, caption='未来 50 步 · 六轴预测 · 5 秒'),
                plot(s['trace'], caption='最近 4 秒 · 六轴实际反馈'),
                table(q, pred[0] if pred is not None else None, ('当前反馈', '预测首步')),
                s['logs'], helptext, details, times, *updates, home_label,
                table(q, s['home'], ('当前实际', '起始位')))

    def preview(task, prompt, end_mode, duration, chunk, speed, gripper_speed, request: gr.Request):
        return checked(engine.start, 'preview', task, prompt, 0 if end_mode == 'manual' else duration, chunk, speed, request.session_hash, gripper_speed)

    def execute(task, prompt, end_mode, duration, chunk, speed, gripper_speed, request: gr.Request):
        return checked(engine.start, 'executing', task, prompt, 0 if end_mode == 'manual' else duration, chunk, speed, request.session_hash, gripper_speed)

    def prepare(ack, request: gr.Request):
        return checked(engine.prepare, ack, request.session_hash)

    def return_home(ack, request: gr.Request):
        return checked(engine.go_home, ack, request.session_hash)

    def replay(name, index):
        row, images, count, index = checked(engine.replay, name, index)
        title = f'第 {index + 1} / {count} 次预测 · 往返 {row["latency_ms"]:.0f} ms'
        commands = [[i + 1, *[round(float(v) / 1000, 3) for v in cmd['sent']]]
                    for i, cmd in enumerate(row['sent_commands'])]
        title += f' · 实际下发 {len(commands)} 条'
        return (*images, plot(row['actions'], caption='当时预测的 50 步动作'),
                table(row['state'], row['actions'][0], ('输入 state', '预测首步')),
                gr.update(minimum=0, maximum=max(1, count - 1), value=index, interactive=True), title, commands)

    with gr.Blocks(title='Piper Studio · 可视化推理', analytics_enabled=False, fill_width=True) as app:
        gr.HTML('<header class="studio-header"><div class="brand-symbol">P<span>•</span></div>'
                '<div><div class="brand">PIPER <b>STUDIO</b></div><div class="brand-sub">看见模型的判断，掌握每一次动作。</div></div>'
                '<div class="header-meta"><span>' + ('SIMULATED · DEMO POLICY' if engine.config.mode == 'sim' else 'REALTIME VLA') +
                '</span><span class="version">CONTINUOUS / 03</span></div></header>')
        notice = gr.Markdown('连接设备与模型 → 预览预测 → 准备从臂 → 开始执行。', elem_id='notice')
        with gr.Tabs():
            with gr.Tab('实时推理', id='inference'):
                with gr.Row(elem_classes='capture-layout'):
                    with gr.Column(scale=8, min_width=540):
                        metrics = gr.HTML(status_html(engine.snapshot()))
                        with gr.Row():
                            front = PreviewImage(label='01 / 场景相机', interactive=False, height=300,
                                                 format='jpeg', buttons=['fullscreen'])
                            wrist = PreviewImage(label='02 / 腕部相机', interactive=False, height=300,
                                                 format='jpeg', buttons=['fullscreen'])
                        predict_plot = gr.HTML(plot())
                        timing = gr.Markdown('预测结果将在这里实时更新。', elem_classes='microcopy')
                        with gr.Accordion('实际反馈轨迹与预测首步', open=False):
                            feedback_plot = gr.HTML(plot())
                            predict_table = gr.HTML(table(None, None, ('当前反馈', '预测首步')))
                    with gr.Column(scale=3, min_width=305, elem_classes='control-panel'):
                        gr.HTML('<div class="section-label"><span>01</span> 连接与任务</div>')
                        with gr.Row():
                            connect = gr.Button('连接设备', size='sm', elem_id='infer-connect')
                            model_connect = gr.Button('连接模型', size='sm', elem_id='infer-model')
                        task = gr.Textbox(value='小黄鸭放入笔筒', label='任务名称', lines=1)
                        prompt = gr.Textbox(value=PROMPT, label='模型提示词 · English', lines=2)
                        end_mode = gr.Radio(choices=[('完成后手动停止', 'manual'), ('到时自动停止', 'timed')],
                                            value='manual', label='结束方式', interactive=True)
                        duration = gr.Number(value=180, minimum=5, maximum=3600, precision=0,
                                             label='到时停止 · 秒', interactive=True, visible=False)
                        preview_button = gr.Button('◉  仅预览预测', variant='primary', interactive=False, elem_id='infer-preview')
                        gr.HTML('<p class="microcopy">先检查两路画面与预测。仅预览不会使能或驱动机械臂。</p>')
                        gr.HTML('<div class="section-label"><span>02</span> 从臂执行</div>')
                        ack = gr.Checkbox(value=False, label='主臂已断开，运动范围内无人和障碍物',
                                          interactive=True, elem_id='infer-safety-ack', key='infer-safety-ack')
                        ack_status = gr.Markdown('尚未勾选执行确认', elem_classes='microcopy')
                        prepare_button = gr.Button('使能并保持当前姿态', size='sm', interactive=False, elem_id='infer-prepare')
                        prepared = gr.Markdown('从臂未准备', elem_classes='microcopy')
                        execute_button = gr.Button('▶  开始执行', interactive=False, elem_id='infer-execute')
                        with gr.Row():
                            pause = gr.Button('Ⅱ  暂停', size='sm')
                            stop = gr.Button('■  停止', variant='stop', elem_id='infer-stop')
                        gr.HTML('<p class="microcopy">停止会清空预测并保持当前位置，不解除电机使能。紧急情况使用机械臂硬件急停。</p>')
                        home_button = gr.Button('↶  复位到起始位', interactive=False, elem_id='infer-home')
                        gr.Markdown('到位标准：六轴误差 ≤ 2.5°、夹爪误差 ≤ 1 mm。完成后显示实际误差。', elem_classes='microcopy')
                        with gr.Accordion('起始位设置', open=False):
                            remember_button = gr.Button('记住当前为起始位', size='sm', interactive=False)
                            home_label = gr.Markdown('尚未设置起始位', elem_classes='microcopy')
                            home_table = gr.HTML(table(None, None, ('当前实际', '起始位')))
                        speed = gr.Slider(1, 15, value=5, step=1, label='关节速度上限 · °/s', interactive=True)
                        gripper_speed = gr.Slider(10, 100, value=engine.config.max_gripper_speed_mm_s,
                            step=5, label='夹爪速度上限 · mm/s', interactive=True)
                        gr.Markdown('两项速度在下一次开始时生效。夹爪行程仍限制在 0–70 mm。', elem_classes='microcopy')
                        with gr.Accordion('执行参数', open=False):
                            chunk = gr.Slider(1, 10, value=5, step=1, label='重新预测间隔 · 0.1 秒 / 步', interactive=True)
                            url = gr.Textbox(value='ws://127.0.0.1:8000', label='模型地址 · SSH 隧道')
                            gr.Markdown('参数在下一次开始时生效。默认每 0.5 秒重新观察；后台预测时继续执行有效轨迹，以 50 Hz 插值并平滑衔接。模型或轨迹超时仍会停止。')
                        gr.HTML('<div class="section-label"><span>03</span> 关节监视</div>')
                        live_table = gr.HTML(table(None, None, ('从臂实际', '实际下发')))
                with gr.Accordion('运行记录与诊断', open=False):
                    record_path = gr.Markdown('尚无本次运行')
                    logs = gr.Textbox(lines=6, interactive=False, label='最近事件')
                    with gr.Row():
                        disconnect = gr.Button('断开设备通信', size='sm')
                        inspect = gr.Button('查看详细连接状态', size='sm')
                    diagnostic = gr.JSON(label='连接诊断')
            with gr.Tab('运行回看', id='review'):
                gr.Markdown('回看保存的模型输入和预测，可逐次检查。本页只显示历史记录。')
                with gr.Row():
                    runs = gr.Dropdown(choices=engine.runs(), label='选择运行', value=None, scale=7)
                    reload_runs = gr.Button('刷新记录', scale=1)
                    load_run = gr.Button('载入', variant='primary', scale=1)
                with gr.Row():
                    past_front = PreviewImage(label='当时的场景画面', height=340, interactive=False, format='jpeg')
                    past_wrist = PreviewImage(label='当时的腕部画面', height=340, interactive=False, format='jpeg')
                past_status = gr.Markdown('请选择一次运行')
                scrub = gr.Slider(0, 1, value=0, step=1, label='第几次预测 · 从 0 计数', interactive=False)
                past_plot = gr.HTML(plot())
                past_table = gr.HTML(table(None, None, ('输入 state', '预测首步')))
                past_commands = gr.Dataframe(headers=['步', 'J1 °', 'J2 °', 'J3 °', 'J4 °', 'J5 °', 'J6 °', '夹爪 mm'],
                                              label='本段实际下发 · 限速与限幅后', interactive=False, value=[])
            with gr.Tab('使用说明', id='help'):
                gr.Markdown('''
### 使用顺序

1. 保持 RealSense 彩色取图终端与 SSH 隧道运行，点击“连接设备”“连接模型”。四个连接标识应变绿。
2. 核对场景 / 腕部两路画面的顺序，填写任务名称与提示词，点击“仅预览预测”。此时“下发动作”为 0。
3. 点击暂停；确认主臂已断开、运动范围清空，勾选确认，点击“使能并保持当前姿态”。
4. 等待准备完成，点击“开始执行”。真机默认关节 5°/s、夹爪 50 mm/s，夹爪行程 0–70 mm。两项速度可单独调整，在下一次开始时生效。以 50 Hz 连续下发，每 0.5 秒从新观测预测，后台推理不阻塞控制。
5. 默认由你在任务完成后点击停止，不再固定 60 秒结束。也可选择到时自动停止并填写秒数；到时停止不代表任务完成。
6. 点击“复位到起始位”返回本轮执行前的姿态。首次执行自动记住起始位；也可在停止时手动记住当前姿态。复位成功后可以直接开始下一轮。

### 运行中断

页面应保持在前台。每次新预测须在 1.5 秒内就绪；就绪后保留最多 2 秒的未来轨迹，并提前请求下一段。新请求超时或轨迹耗尽会停止。页面心跳中断 3 秒、相机 / CAN 超时、控制周期超过 100 ms、关节限位或跟随误差异常也会停止。连接恢复后不会自动恢复运动。

平台沿用高跟随模式（0xAD），在电脑端限制每步变化；不自动 ResetPiper 或标定零点。页面停止依赖电脑和 CAN，不能代替硬件急停。

### 模型与记录

模拟模式使用演示策略；真机模式连接兼容的 OpenPI WebSocket 服务。请使用与训练时一致的配置、数据变换和归一化参数，具体部署步骤见仓库 server/README.md。

模型输出是 50 × 7 的绝对目标：六轴 + 夹爪。输入 state 使用上一次实际发送的命令，首次预览由新鲜反馈初始化。原始预测和限速后的命令分别保存。

夹爪输出统一限制到 0–70 mm；90–100 mm 的预测仍会继续执行，只发送最多 70 mm。关节越界或非有限数值继续阻止执行。

每次运行保存 meta.json、events.jsonl 和每次预测的两张输入 JPEG，默认目录 runtime/inference-runs。运行回看显示这些输入快照及预测；原始 duck 训练数据不会写入本次推理日志。
''')
                gr.Textbox(value=str(engine.run_root), label='记录保存位置', interactive=False)
        global_stop = gr.Button('■ 停止执行', variant='stop', elem_id='global-infer-stop')
        gr.HTML('<div class="studio-footer"><span>PIPER STUDIO / INFERENCE</span><span>REAL SENSE × 2 / 6 AXES + GRIPPER</span><span>LOCAL CONTROL · REMOTE POLICY</span></div>')

        connect.click(lambda: checked(engine.connect), outputs=notice, api_name='infer_connect', concurrency_id='connection')
        model_connect.click(lambda u: checked(engine.check_model, u), inputs=url, outputs=notice, api_name='infer_model', concurrency_id='connection')
        inputs = [task, prompt, end_mode, duration, chunk, speed, gripper_speed]
        end_mode.change(lambda v: gr.update(visible=v == 'timed'), inputs=end_mode, outputs=duration, api_name=False, queue=False)
        ack.input(lambda value: '✓ 已勾选，可准备执行或复位' if value else '尚未勾选执行确认',
                  inputs=ack, outputs=ack_status, api_name='infer_ack_status', queue=False)
        preview_button.click(preview, inputs=inputs, outputs=notice, api_name='infer_preview')
        execute_button.click(execute, inputs=inputs, outputs=notice, api_name='infer_execute')
        prepare_button.click(prepare, inputs=ack, outputs=notice, api_name='infer_prepare')
        remember_button.click(lambda: checked(engine.remember_home), outputs=notice, api_name='infer_remember_home')
        home_button.click(return_home, inputs=ack, outputs=notice, api_name='infer_home')
        pause.click(lambda: engine.stop(pause=True), outputs=notice, api_name='infer_pause', queue=False)
        stop.click(engine.stop, outputs=notice, api_name='infer_stop', queue=False)
        global_stop.click(engine.stop, outputs=notice, api_name='infer_global_stop', queue=False)
        disconnect.click(lambda: checked(engine.disconnect), outputs=notice, api_name='infer_disconnect', concurrency_id='connection')
        inspect.click(lambda: {'config': engine.config.to_dict(), 'model': engine.model,
                              'can': engine.robot.diagnostics() if engine.robot and hasattr(engine.robot, 'diagnostics') else []},
                      outputs=diagnostic, api_name='infer_diagnostics')
        reload_runs.click(lambda: gr.update(choices=engine.runs()), outputs=runs, api_name='infer_runs')
        replay_outputs = [past_front, past_wrist, past_plot, past_table, scrub, past_status, past_commands]
        load_run.click(lambda n: replay(n, 0), inputs=runs, outputs=replay_outputs, api_name='infer_load_run')
        scrub.input(replay, inputs=[runs, scrub], outputs=replay_outputs, api_name='infer_seek', queue=False)
        gr.Timer(0.25).tick(refresh, outputs=[front, wrist, metrics, live_table, predict_plot, feedback_plot,
            predict_table, logs, prepared, record_path, timing, preview_button, execute_button, prepare_button,
            model_connect, connect, remember_button, home_button, home_label, home_table],
            queue=False, show_progress='hidden', api_name='infer_status')
    return app


def main():
    parser = argparse.ArgumentParser(description='Piper Studio 可视化推理平台')
    parser.add_argument('--config', type=Path, default=Path('configs/mac_inference.json'))
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--connect', action='store_true', help='只连接设备读取反馈，不使能或运动')
    parser.add_argument('--sim', action='store_true')
    args = parser.parse_args()
    cfg = Config() if args.sim else Config.load(args.config)
    engine = InferenceEngine(cfg, 'runtime/sim-inference-runs' if args.sim else 'runtime/inference-runs')
    atexit.register(engine.close)
    if args.connect:
        try:
            engine.connect()
        except Exception as exc:
            engine.log('自动连接失败，请在页面重试：' + str(exc))
    try:
        engine.check_model('ws://127.0.0.1:8000')
    except Exception as exc:
        engine.log('模型尚未连接：' + str(exc))
    app = build_app(engine)
    theme = gr.themes.Base(primary_hue='orange', secondary_hue='teal', neutral_hue='slate',
                           font=['Inter', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'])
    try:
        app.queue(default_concurrency_limit=1, max_size=16).launch(server_name='127.0.0.1', server_port=args.port,
            inbrowser=args.open, share=False, theme=theme,
            css_paths=[Path(__file__).with_name('styles.css'), Path(__file__).with_name('inference.css')],
            js="() => { document.body.classList.add('dark'); }", footer_links=[], show_error=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
