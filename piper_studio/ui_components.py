"""Shared platform UI rendering, extracted without behavior changes."""
import html
import numpy as np

COLORS = ["#61d6b5", "#f7aa65", "#73aafa", "#c5a4fa", "#f28099", "#d2d97b"]

def esc(value):
    return html.escape(str(value))

def table(first, second, labels=("从臂反馈", "主臂目标"), raw=False, stale_channels=()):
    def fmt(values, i):
        if values is None or not np.isfinite(values[i]):
            return "—"
        return f"{float(values[i]):.0f}" if raw else f"{float(values[i]) / 1000:.2f}"
    rows = "".join(f"<tr><td>{'J' + str(i + 1) if i < 6 else '夹爪'}</td>"
                   f"<td>{fmt(first, i)}</td><td class='{'stale-target' if i in stale_channels else ''}'>{fmt(second, i)}</td>"
                   f"<td>{'0.001°' if raw and i < 6 else 'µm' if raw else '°' if i < 6 else 'mm'}</td></tr>"
                   for i in range(7))
    return (f'<div class="signal-table"><table><thead><tr><th>关节</th><th>{esc(labels[0])}</th>'
            f'<th>{esc(labels[1])}</th><th>单位</th></tr></thead><tbody>{rows}</tbody></table></div>')

def plot(values=None, cursor=None, caption="六轴目标轨迹 · °"):
    w, h, left, top, bottom = 760, 175, 40, 15, 145
    if values is None or len(values) < 2:
        return '<div class="plot-empty">连接设备后显示关节轨迹</div>'
    a = np.asarray(values, dtype=float)[:, :6] / 1000
    if len(a) > 300:
        indices = np.linspace(0, len(a) - 1, 300).astype(int)
        a = a[indices]
    low, high = float(a.min()), float(a.max())
    margin = max((high - low) * 0.08, 5)
    low, high = low - margin, high + margin
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="六个关节的角度轨迹">']
    for v in np.linspace(low, high, 4):
        y = bottom - (v - low) / (high - low) * (bottom - top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w - 10}" y2="{y:.1f}" stroke="#233345"/>'
                     f'<text x="2" y="{y + 4:.1f}" fill="#8b9db0" font-size="10">{v:.0f}°</text>')
    for j, color in enumerate(COLORS):
        pts = " ".join(f"{left + i / (len(a) - 1) * (w - left - 10):.1f},"
                       f"{bottom - (v - low) / (high - low) * (bottom - top):.1f}" for i, v in enumerate(a[:, j]))
        parts.append(f'<polyline points="{pts}" stroke="{color}" fill="none" stroke-width="1.7"/>')
        parts.append(f'<text x="{left + j * 52}" y="170" fill="{color}" font-size="11">J{j + 1}</text>')
    if cursor is not None:
        x = left + max(0, min(1, cursor)) * (w - left - 10)
        parts.append(f'<line x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="150" stroke="#fff" stroke-dasharray="4 4"/>')
    parts.append('</svg>')
    return f'<div class="trajectory"><div class="eyebrow">{esc(caption)}</div>{"".join(parts)}</div>'
