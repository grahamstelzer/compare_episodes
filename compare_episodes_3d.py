"""
Interactive 3D comparison of reference episodes and rollout episodes.

Page layout (generated HTML):
  * the 3D plot fills the window
  * right, under the LINES panel: a small KEY (line colors, start/end dots) with a "More info"
    section (SO101 diagram with the plotted joints highlighted, axis meaning,
    lowest points, joint ranges)
  * top-right: a scrollable LINES panel to hide / isolate / select groups of lines
  * bottom-right (only with --video_key): a small EPISODE VIDEO window. Pick any
    reference / rollout episode from the dropdowns, or click a line in the plot to
    load that episode and jump to the time of the point you clicked. Once an episode
    is loaded, "preview frame on hover" scrubs that video as the cursor follows the
    episode's line (off by default, so hovering never loads other videos). A yellow dot on the loaded
    episode's line follows the video's current frame.
  * bottom-center (only with --joints): a 2D JOINTS-OVER-TIME graph for the trajectory
    clicked last. Header click hides it; "overlay mode" keeps earlier clicks on screen.
Colors follow the system light/dark setting live.

Requires: pip install plotly   (plus lerobot; and placo if you use --urdf;
                                ffmpeg on PATH only if you use --video_mode clips)

The page (HTML, CSS, JavaScript) lives in compare_episodes_3d.template.html, which must sit in the
same folder as this script. It is filled in at run time, so the HTML output is still one
self-contained file.

Episode selection (works for --ref_episodes and --run_episodes):
  0 1 2            individual episodes
  0-9              inclusive range
  0-20:2           range with step (every 2nd episode)
  5-               from episode 5 to the last one
  0-4,8,12-15      comma-separated mix
  all              every episode

3D axes (pick one):
  (default)                        first 3 principal components of ALL reference episodes
  --urdf PATH --units degrees      gripper xyz via LeRobot's RobotKinematics + the SO101 URDF
  --urdf PATH --units normalized --calib CALIB.json
                                   same, converting -100..100 values to degrees first
  --fk_module my_fk.py             gripper xyz from your own fk(q) -> (N, 3)

Joint graph (optional, works with any 3D axes above):
  --joints 0 2 3                   joint columns to graph over time, by index
  --joints shoulder_pan wrist_flex same, by name (names come from the dataset)

Videos (optional):
  --video_key observation.images.camera3
                                   camera to show in the corner window (looked up in BOTH datasets)
  --video_mode link (default)      the page points at the dataset's own mp4 files (relative paths),
                                   using each episode's from/to timestamps inside the file
  --video_mode clips               ffmpeg cuts one small H.264 clip per episode into <out>_videos/.
                                   Use this if the browser can't play the dataset's codec (LeRobot
                                   often encodes AV1, which Safari/older browsers can't decode) or
                                   if you want to move/share the HTML.

Example:
  python compare_episodes_3d.py \
      --ref_repo <training_dataset> --ref_episodes 0-19 \
      --run_repo grahamwichhh/onlyhil --run_episodes 0-4 \
      --urdf so101_new_calib.urdf --units degrees \
      --video_key observation.images.camera3 --out cmp3d.html
"""
import argparse
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np
import plotly.graph_objects as go
import torch

# from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# "canonical" joint names
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
CANON = ARM_JOINTS + ["gripper"]

# colorscales chosen to stay visible on BOTH white and near-black backgrounds.
BLUES = [[0, "#a6cee3"], [1, "#1f78b4"]]
REDS = [[0, "#fcae91"], [1, "#cb181d"]]


# the template.html page has the main HTML, CSS, and JS
#  path is resolved relative to this file
#  uses placeholders (__PLOT__, __INFO__,__EPS__, __FPS__) that running this python file fills
#  video-panel section is removed when no video is available
#  accepts either file name: compare_episodes_3d.template.html or compare_episodes_3d_template.html
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [os.path.join(_HERE, n) for n in
               ("compare_episodes_3d.template.html", "compare_episodes_3d_template.html")]
TEMPLATE_PATH = next((c for c in _CANDIDATES if os.path.exists(c)), _CANDIDATES[0])


# ---------------------------------------------------------------- selection
def parse_selection(tokens, total):
    """
    compiles list of episodes for dataset load param
    remember inputs are like     "all"     "0-99:5"     "13, 23, 24"
    NOTE: uses "tokens" variable name since we are technically parsing strings
    """
    eps = []
    for tok in tokens:
        for part in tok.split(","):

            part = part.strip()
            if not part:
                continue

            if part == "all":
                eps += list(range(total))
                continue

            step = 1
            if ":" in part:
                part, s = part.split(":")
                step = int(s)

            if "-" in part:
                a, b = part.split("-")
                a = int(a)
                b = int(b) if b else total - 1
                eps += list(range(a, b + 1, step))
            else:
                eps.append(int(part))

    return sorted(set(eps)) # should return like [0, 1, 2, 45, 89, ...]


def joint_names_from(ds, col, n):
    try:
        names = ds.features[col]["names"]
        if isinstance(names, dict):
            names = next(iter(names.values()))
        names = [str(x).replace(".pos", "") for x in names]
        if len(names) == n:
            return names
    except Exception:
        pass
    return [f"j{i}" for i in range(n)]


def resolve_joints(tokens, names):
    """Turn --joints values (column indices like 2, or names like elbow_flex) into column indices."""
    sel = []
    for tok in tokens:
        if tok.lstrip("-").isdigit():
            k = int(tok)
            if not 0 <= k < len(names):
                raise SystemExit(f"--joints: index {k} is out of range 0..{len(names) - 1}; joints: {names}")
        elif tok in names:
            k = names.index(tok)
        else:
            raise SystemExit(f"--joints: '{tok}' is not a joint name; joints: {names}")
        if k not in sel:
            sel.append(k)
    return sel


# ---------------------------------------------------------------- videos
def find_videos(ds, key, episodes):
    """{episode_index: {"path", "start", "end"}} for episodes that have a video under `key`.

    LeRobot v3 packs many episodes into one mp4 (videos/<key>/chunk-XXX/file-XXX.mp4) and records each
    episode's slice as videos/<key>/from_timestamp and to_timestamp in the episodes metadata. Older
    datasets have one mp4 per episode, so start = 0 and the end falls back to the episode length."""
    keys = getattr(ds.meta, "video_keys", None)
    if keys is not None and key not in keys:
        print(f"warning: video key '{key}' not found in {ds.repo_id}; available: {list(keys)}")
        return {}
    out = {}
    for e, q in episodes.items():
        try:
            path = os.path.join(str(ds.root), str(ds.meta.get_video_file_path(e, key)))
            row = ds.meta.episodes[e]
            start = float(row.get(f"videos/{key}/from_timestamp", 0.0) or 0.0)
            end = row.get(f"videos/{key}/to_timestamp")
            end = float(end) if end is not None else start + len(q) / ds.fps
        except Exception as exc:
            print(f"warning: could not resolve video for {ds.repo_id} ep{e}: {exc}")
            continue
        if not os.path.exists(path):
            print(f"warning: video file missing: {path}")
            continue
        out[e] = {"path": path, "start": start, "end": end}
    return out


def make_clips(vids, out_dir, tag):
    """Cut each episode into its own H.264 clip (browser-friendly) with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("warning: ffmpeg not found on PATH; falling back to --video_mode link")
        return vids
    os.makedirs(out_dir, exist_ok=True)
    clipped = {}
    for e, v in sorted(vids.items()):
        dst = os.path.join(out_dir, f"{tag}_ep{e:06d}.mp4")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{v['start']:.4f}",
               "-t", f"{v['end'] - v['start']:.4f}", "-i", v["path"], "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast",
               "-movflags", "+faststart", dst]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"warning: ffmpeg failed for {tag} ep{e}: {exc}")
            continue
        clipped[e] = {"path": dst, "start": 0.0, "end": v["end"] - v["start"]}
        print(f"  clip {dst}")
    return clipped


def video_url(path, out_html):
    """Path relative to the HTML file (works on file:// and when served from a common parent)."""
    try:
        rel = os.path.relpath(path, os.path.dirname(os.path.abspath(out_html)))
        return urllib.parse.quote(rel.replace(os.sep, "/"))
    except ValueError:  # different drive on Windows
        return pathlib.Path(path).resolve().as_uri()


def load_episodes(repo, root, tokens, source, video_key=None):
    """Returns ({episode_index: array (T, A)}, fps, joint_names, {episode_index: video info})."""

    # user specifies episodes like all, 0-99:5 etc. 
    # either way, need total to parse selection, full dataset load is expensive so get meta only:
    total = LeRobotDatasetMetadata(repo, root=root).total_episodes
    eps = parse_selection(tokens, total)

    ds = LeRobotDataset(repo, root=root, episodes=eps)
    col = "action" if source == "action" else "observation.state"

    data = torch.stack(list(ds.hf_dataset[col])).numpy()
    # ep_idx = np.asarray([int(x) for x in ds.hf_dataset["episode_index"]])
    # out = {e: data[ep_idx == e] for e in eps if (ep_idx == e).any()}
    # missing = [e for e in eps if e not in out]

    # Convert episode IDs from the dataset into a NumPy integer array so we can
    # match each recorded timestep to its source episode efficiently.
    episode_indices = np.asarray([int(x) for x in ds.hf_dataset["episode_index"]], dtype=int)

    # Build a dictionary mapping each selected episode number to the subset of
    # trajectory data belonging to that episode. We iterate over the requested
    # episodes in order and keep only the rows whose episode index matches.
    out = {}
    for episode in eps:
        matches = episode_indices == episode
        if matches.any():
            out[episode] = data[matches]

    missing = [episode for episode in eps if episode not in out]

    if missing:
        print(f"warning: episodes not found in {repo}: {missing}")
        
    vids = find_videos(ds, video_key, out) if video_key else {}
    return out, ds.fps, joint_names_from(ds, col, data.shape[1]), vids


# ---------------------------------------------------------------- kinematics
def load_fk_module(path):
    spec = importlib.util.spec_from_file_location("user_fk", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fk


def strip_geometry(urdf_path):
    """Write a copy of the URDF without <visual>/<collision> so no STL meshes are needed.
    Forward kinematics only uses joint origins and axes, not the meshes."""
    tree = ET.parse(urdf_path)
    for link in tree.getroot().iter("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    out = os.path.join(tempfile.gettempdir(), "so101_kinematics_only.urdf")
    tree.write(out)
    return out


def make_urdf_fk(urdf_path, names, units, calib_path, keep_geometry):
    from lerobot.model.kinematics import RobotKinematics

    urdf = os.path.abspath(urdf_path) if keep_geometry else strip_geometry(urdf_path)
    kin = RobotKinematics(urdf_path=urdf, target_frame_name="gripper_frame_link",
                          joint_names=ARM_JOINTS)
    idx = [names.index(n) for n in ARM_JOINTS] if all(n in names for n in ARM_JOINTS) else list(range(5))

    scale = None
    if units == "normalized":
        # UNVERIFIED: assumes LeRobot's RANGE_M100_100 mode and a calibration JSON shaped like
        # {"shoulder_pan": {"range_min": .., "range_max": .., "drive_mode": 0, ...}, ...}
        calib = json.load(open(calib_path))
        lo = np.array([calib[n]["range_min"] for n in ARM_JOINTS], float)
        hi = np.array([calib[n]["range_max"] for n in ARM_JOINTS], float)
        drive = np.array([-1.0 if calib[n].get("drive_mode", 0) else 1.0 for n in ARM_JOINTS])
        scale = drive * ((hi - lo) / 2.0) / 100.0 * 360.0 / 4095.0

    def fk(q):
        q5 = np.asarray(q, float)[:, idx]
        if scale is not None:
            q5 = q5 * scale
        out = np.empty((len(q5), 3))
        for i, row in enumerate(q5):
            out[i] = kin.forward_kinematics(row)[:3, 3]
        return out

    return fk


# ---------------------------------------------------------------- plotting
def add_traj(fig, xyz, q, names, label, colorscale, fps, opacity):
    """Adds 3 traces (line, start marker, end marker); returns their indices."""
    first = len(fig.data)
    t = np.linspace(0, 1, len(xyz))
    secs = (np.arange(len(xyz)) / fps)[:, None]
    custom = np.hstack([secs, q])
    tmpl = label + "<br>t = %{customdata[0]:.3f} s<br>" + "<br>".join(
        f"{n}: %{{customdata[{i + 1}]:.1f}}" for i, n in enumerate(names)
    ) + "<extra></extra>"
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=label,
        showlegend=False, opacity=opacity, customdata=custom, hovertemplate=tmpl,
        line=dict(color=t, colorscale=colorscale, width=5, cmin=-0.3, cmax=1.0),
    ))
    for pt, color, tag in [(xyz[0], "#2ecc71", "start"), (xyz[-1], "#ff8c00", "end")]:
        fig.add_trace(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers", showlegend=False,
            marker=dict(size=5, color=color, line=dict(color="#888888", width=1)),
            name=f"{label} {tag}", hoverinfo="name"))
    return [first, first + 1, first + 2]


# ---------------------------------------------------------------- SO101 diagram
JPOS = {"shoulder_pan": (65, 204), "shoulder_lift": (65, 160), "elbow_flex": (120, 62),
        "wrist_flex": (232, 88), "wrist_roll": (246, 140), "gripper": (256, 165)}
JLABEL = {"shoulder_pan": (92, 207, "start"), "shoulder_lift": (78, 168, "start"),
          "elbow_flex": (120, 46, "middle"), "wrist_flex": (238, 72, "middle"),
          "wrist_roll": (262, 138, "start"), "gripper": (272, 162, "start")}
TIP = (266, 222)


def star_points(cx, cy, ro=7.5, ri=3.2):
    pts = []
    for k in range(10):
        r = ro if k % 2 == 0 else ri
        a = math.radians(-90 + 36 * k)
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    return " ".join(pts)


def arm_svg(spec):
    """Schematic side view of the SO101 (not to scale) with plotted joints highlighted.
    spec = {"highlight": {canonical_joint: [tags]}, "tip": bool}"""
    hl, tip = spec["highlight"], spec["tip"]
    p = ['<svg viewBox="0 0 340 252" class="arm" role="img" aria-label="SO101 arm schematic">']
    # links
    p.append('<rect class="linkfill" x="25" y="220" width="80" height="12" rx="3"/>')
    p.append('<rect class="linkfill" x="45" y="188" width="40" height="34" rx="4"/>')
    for a, b in [((65, 160), (120, 62)), ((120, 62), (232, 88)), ((232, 88), (246, 140)),
                 ((246, 140), (268, 226)), ((256, 165), (290, 212))]:
        p.append(f'<line class="linkline" x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"/>')
    # rotation hints for pan and roll
    p.append(f'<ellipse class="rot{" on" if "shoulder_pan" in hl else ""}" cx="65" cy="204" rx="26" ry="6"/>')
    p.append(f'<ellipse class="rot{" on" if "wrist_roll" in hl else ""}" cx="246" cy="140" rx="15" ry="4" '
             f'transform="rotate(-15 246 140)"/>')
    # joints + labels
    for name, (x, y) in JPOS.items():
        p.append(f'<circle class="j {"on" if name in hl else "off"}" cx="{x}" cy="{y}" r="7"/>')
    for name, (x, y, anchor) in JLABEL.items():
        on = name in hl
        tags = "".join(f' <tspan class="tag">{t}</tspan>' for t in hl.get(name, []))
        p.append(f'<text class="lab {"on" if on else "off"}" x="{x}" y="{y}" text-anchor="{anchor}">{name}{tags}</text>')
    # tracked point
    if tip:
        p.append(f'<line class="lead" x1="{TIP[0]}" y1="{TIP[1] + 7}" x2="{TIP[0] - 6}" y2="{TIP[1] + 17}"/>')
        p.append(f'<polygon class="tip" points="{star_points(*TIP)}"/>')
        p.append(f'<text class="lab tiplab" x="{TIP[0] - 4}" y="246" text-anchor="end">'
                 f'tracked point: gripper_frame_link</text>')
        p.append('<line class="zax" x1="16" y1="76" x2="16" y2="34"/>'
                 '<polygon class="zhead" points="16,28 11,38 21,38"/>'
                 '<text class="lab" x="24" y="40">z (up)</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- info panel
def range_table(refs, runs, names):
    def rng(d):
        allq = np.concatenate(list(d.values()), axis=0)
        return allq.min(0), allq.max(0)

    rmin, rmax = rng(refs)
    nmin, nmax = rng(runs)
    rows = "".join(
        f"<tr><td>{n}</td><td>{rmin[i]:.0f}</td><td>{rmax[i]:.0f}</td>"
        f"<td>{nmin[i]:.0f}</td><td>{nmax[i]:.0f}</td></tr>"
        for i, n in enumerate(names)
    )
    return ("<table><tr><th>joint</th><th>ref min</th><th>ref max</th>"
            "<th>run min</th><th>run max</th></tr>" + rows + "</table>")


def extremes(trajs, labels, names, fps):
    items = []
    for k, lab in enumerate(labels):
        best = None
        for tag, xyz, q in trajs:
            i = int(np.argmin(xyz[:, k]))
            if best is None or xyz[i, k] < best[0]:
                best = (xyz[i, k], tag, i, q[i])
        v, tag, i, qv = best
        joints = ", ".join(f"{n}={x:.0f}" for n, x in zip(names, qv))
        items.append(f"<li><b>{lab}</b>: lowest {v:.2f} in <b>{tag}</b> at {i / fps:.3f}s "
                     f"<span class='muted'>({joints})</span></li>")
    return "<ul>" + "".join(items) + "</ul>"


def axis_explanation(fk_used, units):
    if fk_used:
        conv = ("Joint values were converted from the dataset's normalized units to degrees using your "
                "calibration file first." if units == "normalized" else
                "Joint values were taken as degrees.")
        return ("Axes are the <b>gripper position in meters</b> in the URDF's base frame (z is up), from "
                "forward kinematics. Negative = the other side of the base origin. " + conv)
    return ("Each axis is a <b>principal component</b>, not a joint or a position: the reference mean pose "
            "is subtracted and the result is projected onto the 3 directions where reference data varies "
            "most. Axis titles give variance share and driving joints. <b>Negative = opposite side of the "
            "average pose</b>; &minus;50 = 50 units from average along that direction.")


def diagram_caption(spec, args):
    if spec["kind"] == "fk":
        who = "forward kinematics (the five highlighted arm joints)" if args.urdf else \
              "your fk() function (the highlighted joints are an assumption)"
        return (f"Gripper position is computed by {who}. The star marks the tracked point, "
                "<code>gripper_frame_link</code>: on the gripper's axis near the jaw tips, about 10&nbsp;cm "
                "from the wrist-roll joint (the URDF's <code>gripper_frame_joint</code> offset). "
                "Jaw opening is not used. Schematic, not to scale; star position approximate.")
    return ("Every joint contributes to every axis. Highlighted joints are the two strongest contributors "
            "to each principal component (tags PC1 to PC3). Schematic, not to scale.")


def build_info(args, refs, runs, names, labels, trajs, fps, fk_used, spec, has_video, jnames):
    # src = ("<b>action</b> = commanded target (leader arm in human demos, policy output in rollouts)"
    #        if args.source == "action" else "<b>state</b> = measured follower position")
    key = f"""
<div class="key">
  <div><span class="bar bar-ref"></span> Reference <span class="muted">({len(refs)} ep, {args.ref_source})</span>
       <code>{args.ref_repo}</code></div>
  <div><span class="bar bar-run"></span> Rollout <span class="muted">({len(runs)} ep, {args.run_source})</span>
       <code>{args.run_repo}</code></div>
  <div class="muted">Each line goes <b>pale &rarr; deep</b> from the start to the end of an episode.</div>
  <div><span class="dot dot-start"></span> start of episode &nbsp;
       <span class="dot dot-end"></span> end of episode</div>
</div>"""
    video_help = ""
    if has_video:
        video_help = f"""
<h4>Episode video (bottom right)</h4>
<p>Shows <code>{args.video_key}</code>. Choose Reference or Rollout and an episode to load it, or
<b>click a line</b> in the plot to load that episode and jump to the time of the point you clicked. The
slider covers just that episode. S / M / L resizes the window; click its title bar to collapse it.</p>
<p><b>Yellow dot:</b> marks where the loaded episode is along its line in the plot, at the current video
frame. Play or scrub the video and the dot moves along the trajectory. It hides with its line. When two trajectories are being compared in the joints graph, each gets a dot,
both driven by the same video time.</p>
<p><b>Preview frame on hover:</b> after an episode is loaded, tick this box to scrub that video as the
cursor follows <i>that episode's</i> line in the plot. It is disabled until an episode is loaded and ignores
all other lines, so hovering never loads another video. Hovering also pauses playback.</p>"""
    joint_help = ""
    if jnames:
        joint_help = f"""
<h4>Joints over time (bottom center)</h4>
<p>Graphs <code>{", ".join(jnames)}</code> against time for the line you <b>clicked last</b>. Tick
<b>overlay mode</b> to compare two trajectories (a third click is refused with a warning; dashes change per trajectory,
colors per joint; the yellow dot follows both). A yellow vertical line on the graph marks the video's
current time, in step with the dots;
<b>Clear</b> empties the graph. Click the panel's title bar to hide it. Values are in dataset units.</p>"""
    more = f"""
<details id="more">
<summary>More info</summary>
<p><b>One line = one episode.</b></p>
{arm_svg(spec)}
<p class="muted">{diagram_caption(spec, args)}</p>
<h4>Axes</h4>
<p>{axis_explanation(fk_used, args.units)}</p>
<h4>Using the Lines panel (top right)</h4>
<p>Click a row to hide or show it; double-click to isolate it (double-click again to restore).
Type episode numbers such as <code>0-9</code>, <code>3, 7, 12-</code> or <code>0-20:2</code> and press
<b>Show</b>, <b>Hide</b> or <b>Only</b> (Only hides everything else, in both groups).
Hover a point on a line for time and every joint value.</p>{video_help}{joint_help}
<h4>Rotate button (top left)</h4>
<p>Slowly spins the plot around its vertical axis, starting from the view you set and keeping your
tilt and zoom. Dragging, scrolling or touching the plot stops it, so you can reposition and press
Rotate again.</p>
<h4>Lowest point on each axis</h4>
{extremes(trajs, labels, names, fps)}
<h4>Joint ranges (plotted episodes)</h4>
{range_table(refs, runs, names)}
<p class="muted">Rollouts recorded with record_autonomous=True include your interventions, so a big
excursion in a red line may be you, not the policy.</p>
</details>"""
    return "<summary>Key</summary>" + key + more


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--ref_repo", required=True)
    ap.add_argument("--ref_episodes", nargs="+", default=["0"])
    ap.add_argument("--ref_root", default=None)
    ap.add_argument("--run_repo", required=True)
    ap.add_argument("--run_episodes", nargs="+", default=["0"])
    ap.add_argument("--run_root", default=None)
    ap.add_argument("--source", choices=["action", "state"], default="action",
                        help="default for both datasets")
    ap.add_argument("--ref_source", choices=["action", "state"], default=None,
                    help="overrides --source for the reference dataset")
    ap.add_argument("--run_source", choices=["action", "state"], default=None,
                    help="overrides --source for the rollout dataset")
    ap.add_argument("--joints", nargs="+", default=None, metavar="J",
                    help="joints to graph over time in a 2D panel: column indices or names, "
                         "e.g. 0 2 3 or shoulder_pan wrist_flex")
    ap.add_argument("--urdf", default=None, help="path to so101_new_calib.urdf")
    ap.add_argument("--units", choices=["degrees", "normalized"], default=None,
                    help="units of your dataset's joint values (required with --urdf)")
    ap.add_argument("--calib", default=None, help="calibration JSON (needed for --units normalized)")
    ap.add_argument("--keep_geometry", action="store_true",
                    help="use the URDF as-is (needs its assets/ meshes next to it)")
    ap.add_argument("--fk_module", default=None)
    ap.add_argument("--video_key", default=None,
                    help="camera key to show in the corner video window, e.g. observation.images.camera3")
    ap.add_argument("--video_mode", choices=["link", "clips"], default="link",
                    help="link: point the page at the dataset's mp4 files; "
                         "clips: cut per-episode H.264 clips with ffmpeg next to the HTML")
    ap.add_argument("--out", default="cmp3d.html")
    args = ap.parse_args()

    # override for flags for showing predicted action vs resulting motor state
    args.ref_source = args.ref_source or args.source
    args.run_source = args.run_source or args.source

    if args.urdf and not args.units:
        ap.error("--urdf requires --units degrees|normalized (check your data's min/max first)")
    if args.units == "normalized" and not args.calib:
        ap.error("--units normalized requires --calib <calibration .json>")

    # read the template up front so a missing file fails before the slow dataset loading
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()





    refs, fps, names, ref_vids = load_episodes(args.ref_repo, args.ref_root, args.ref_episodes,
                                               args.ref_source, args.video_key)
    runs, _, _, run_vids = load_episodes(args.run_repo, args.run_root, args.run_episodes,
                                         args.run_source, args.video_key)





    print(f"reference episodes: {sorted(refs)}\nrollout episodes:   {sorted(runs)}")
    print("joint names:", names)
    # validated here (not earlier) because the joint names come from the loaded dataset
    jsel = resolve_joints(args.joints, names) if args.joints else []
    jnames = [names[k] for k in jsel]

    if args.video_key:
        if args.video_mode == "clips":
            clip_dir = os.path.splitext(os.path.abspath(args.out))[0] + "_videos"
            print(f"cutting per-episode clips into {clip_dir}")
            ref_vids = make_clips(ref_vids, clip_dir, "ref")
            run_vids = make_clips(run_vids, clip_dir, "run")
        print(f"videos for '{args.video_key}': {len(ref_vids)} reference, {len(run_vids)} rollout episodes")
    vids_by_group = {"ref": ref_vids, "run": run_vids}

    def canon_of(i):
        n = names[i]
        return n if n in CANON else (CANON[i] if i < len(CANON) else None)

    fk_used = bool(args.urdf or args.fk_module)
    if fk_used:
        if args.urdf:
            fk = make_urdf_fk(args.urdf, names, args.units, args.calib, args.keep_geometry)
        else:
            fk = load_fk_module(args.fk_module)
        project = lambda q: np.asarray(fk(q))
        labels = ["gripper x (m)", "gripper y (m)", "gripper z (m)"]
        spec = dict(kind="fk", tip=True, highlight={n: [] for n in ARM_JOINTS})
    else:
        all_ref = np.concatenate(list(refs.values()), axis=0)
        mu = all_ref.mean(0)
        _, S, Vt = np.linalg.svd(all_ref - mu, full_matrices=False)
        var = S ** 2 / np.sum(S ** 2)
        P = Vt[:3].T
        project = lambda q: (q - mu) @ P
        labels, hl = [], {}
        for k in range(3):
            top = np.argsort(-np.abs(Vt[k]))[:2]
            drivers = ", ".join(f"{'+' if Vt[k][i] > 0 else '−'}{names[i]}" for i in top)
            labels.append(f"PC{k + 1}: {var[k] * 100:.0f}% of variance ({drivers})")
            for i in top:
                c = canon_of(int(i))
                if c:
                    hl.setdefault(c, []).append(f"PC{k + 1}")
        spec = dict(kind="pca", tip=False, highlight=hl)

    opacity = 1.0 if len(refs) + len(runs) <= 4 else 0.7
    fig = go.Figure()
    trajs, meta = [], []
    for group, d, cs in (("ref", refs, BLUES), ("run", runs, REDS)):
        for e, q in sorted(d.items()):
            xyz = project(q)
            label = f"{group} ep{e}"
            traces = add_traj(fig, xyz, q, names, label, cs, fps, opacity)
            trajs.append((label, xyz, q))
            v = vids_by_group[group].get(e)
            video = ({"url": video_url(v["path"], args.out), "start": v["start"], "end": v["end"]}
                     if v else None)
            entry = {"group": group, "ep": int(e), "traces": traces, "video": video}
            if video:
                entry["xyz"] = np.round(xyz, 5).tolist()        # (T, 3) so the page can place the frame dot
            if jsel:
                entry["q"] = np.round(q[:, jsel], 3).tolist()   # (T, len(jsel)) for the 2D graph
            meta.append(entry)

    has_video = any(m["video"] for m in meta)
    if args.video_key and not has_video:
        print("warning: no videos were found, so the video window is omitted")

    if fk_used:
        allxyz = np.concatenate([t[1] for t in trajs], axis=0)
        print("FK sanity check (meters): min", np.round(allxyz.min(0), 3), "max", np.round(allxyz.max(0), 3))
        print("  expect values within roughly +/-0.5 m of the base; much larger => wrong units/conversion")

    fig.update_layout(
        scene=dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2],
                   aspectmode="data"),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
    )

    plot_html = fig.to_html(full_html=False, include_plotlyjs=True, div_id="plot",
                            default_height="100%", default_width="100%",
                            config={"responsive": True})
    info = build_info(args, refs, runs, names, labels, trajs, fps, fk_used, spec, has_video, jnames)
    if not has_video:
        template = re.sub(r"<!-- video-panel:start -->.*?<!-- video-panel:end -->", "", template, flags=re.S)
    if not jsel:
        template = re.sub(r"<!-- joint-panel:start -->.*?<!-- joint-panel:end -->", "", template, flags=re.S)
    values = {"PLOT": plot_html, "INFO": info, "FPS": json.dumps(float(fps)), "EPS": json.dumps(meta),
              "JNAMES": json.dumps(jnames)}
    # single pass: substituted text (plotly.js is ~3 MB) is never re-scanned for placeholders
    html = re.sub(r"__(PLOT|INFO|FPS|EPS|JNAMES)__", lambda m: values[m.group(1)], template)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"saved {args.out} (open in a browser)")
    if has_video and args.video_mode == "link":
        print("  video_mode=link: keep the HTML where its relative video paths still resolve, "
              "or use --video_mode clips for a portable folder")


if __name__ == "__main__":
    main()