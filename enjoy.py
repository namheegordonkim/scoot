import math
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
from imgui_bundle import hello_imgui, imgui, immapp, implot
from matplotlib import colormaps
from torch import nn


def gui(app_state):
    imgui.begin_child("Controls", (imgui.get_content_region_avail().x / 2, 0))
    imgui.push_item_width(220)
    imgui.text("SCOOT: State-Conditioned Shooting (Kim et al., MIG '22)")
    imgui.text_wrapped("One-shot billiards. State: cue ball placement. Action: shot angle. Reward: nearest neighbour among 8.7M precomputed shots.")
    changed, app_state.algo = imgui.combo("Algorithm", app_state.algo, app_state.algos)
    if changed:
        app_state.__dict__.update(app_state.presets[app_state.algo])
    app_state.reset |= changed
    imgui.text_wrapped(app_state.descriptions[app_state.algo])
    changed, app_state.seed = imgui.slider_int("Seed", app_state.seed, 0, 100, flags=imgui.SliderFlags_.always_clamp)
    app_state.reset |= changed
    changed, app_state.n_samples = imgui.slider_int("Samples / iteration", app_state.n_samples, 16, 1024, flags=imgui.SliderFlags_.always_clamp)
    app_state.reset |= changed
    changed, app_state.init_log_std = imgui.slider_float("Initial log std", app_state.init_log_std, -3.0, 1.0, "%.2f", imgui.SliderFlags_.always_clamp)
    app_state.reset |= changed
    changed, app_state.policy_lr = imgui.slider_float("Policy lr", app_state.policy_lr, 1e-6, 1e-2, "%.6f", imgui.SliderFlags_.logarithmic | imgui.SliderFlags_.always_clamp)
    app_state.reset |= changed
    changed, app_state.critic_lr = imgui.slider_float("Critic lr", app_state.critic_lr, 1e-6, 1e-2, "%.6f", imgui.SliderFlags_.logarithmic | imgui.SliderFlags_.always_clamp)
    app_state.reset |= changed
    if app_state.algo >= 2:
        imgui.separator_text("SCOOT features (all off = AWR)")
        changed, app_state.elite = imgui.checkbox("Elite samples: keep A > 0, weight = A", app_state.elite)
        app_state.reset |= changed
        imgui.begin_disabled(app_state.elite)
        changed, app_state.temperature = imgui.slider_float("AWR temperature (beta)", app_state.temperature, 0.05, 5.0, "%.2f", imgui.SliderFlags_.logarithmic | imgui.SliderFlags_.always_clamp)
        app_state.reset |= changed
        imgui.end_disabled()
        changed, app_state.n_heads = imgui.slider_int("Experts (H)", app_state.n_heads, 1, 8, flags=imgui.SliderFlags_.always_clamp)
        app_state.reset |= changed
        changed, app_state.sobol_init = imgui.checkbox("Non-overlapping init (Sobol head offsets)", app_state.sobol_init)
        app_state.reset |= changed
        imgui.begin_disabled(app_state.n_heads == 1)
        changed, app_state.dist_weight = imgui.slider_float("Distance weight (lambda)", app_state.dist_weight, 0.0, 1.0, "%.3f", imgui.SliderFlags_.always_clamp)
        app_state.reset |= changed
        changed, app_state.overlap = imgui.slider_float("Overlap limit d (stds)", app_state.overlap, 0.1, 4.0, "%.2f", imgui.SliderFlags_.always_clamp)
        app_state.reset |= changed
        imgui.end_disabled()
        changed, app_state.curriculum = imgui.checkbox("Curriculum: means -> + std -> + gating", app_state.curriculum)
        app_state.reset |= changed
        imgui.begin_disabled(not app_state.curriculum)
        changed, app_state.std_start = imgui.slider_int("Learn std from iteration", app_state.std_start, 0, 5000, flags=imgui.SliderFlags_.always_clamp)
        app_state.reset |= changed
        changed, app_state.gate_start = imgui.slider_int("Learn gating from iteration", app_state.gate_start, 0, 5000, flags=imgui.SliderFlags_.always_clamp)
        app_state.reset |= changed
        imgui.end_disabled()
    imgui.text_wrapped("Every setting above resets and pauses, so one curve always comes from one configuration.")

    imgui.separator()
    app_state.reset |= imgui.button("Reset")
    imgui.same_line()
    if imgui.button("Pause" if app_state.training else "Run"):
        app_state.training = not app_state.training
    app_state.stage_clicked = False
    for index, name in enumerate(app_state.stage_names):
        imgui.same_line()
        imgui.begin_disabled(index != app_state.stage)
        app_state.stage_clicked |= imgui.button(name)
        imgui.end_disabled()
    changed, app_state.stages_per_frame = imgui.slider_int("Stages / frame (5 = one iteration)", app_state.stages_per_frame, 1, 500, flags=imgui.SliderFlags_.logarithmic | imgui.SliderFlags_.always_clamp)
    changed, app_state.max_iters = imgui.slider_int("Max iterations", app_state.max_iters, 1, 10000, flags=imgui.SliderFlags_.always_clamp)

    if app_state.reset:
        app_state.reset = False
        app_state.training = False
        app_state.stage = 0
        app_state.iteration = 0
        # Rows: samples used, test return, test success rate, batch mean reward.
        app_state.curve = np.zeros((4, 0))
        # Independent streams keep evaluation from changing training.
        app_state.train_rng = torch.Generator(app_state.device).manual_seed(app_state.seed)
        app_state.eval_rng = torch.Generator(app_state.device).manual_seed(app_state.seed + 1)
        app_state.test_s = 2 * torch.rand(2048, 1, generator=app_state.eval_rng, device=app_state.device) - 1
        app_state.grid_s = torch.linspace(-1, 1, 64, device=app_state.device)[:, None]
        # Columns: state, pre-tanh action, action, reward, log-probability at sampling time.
        app_state.buffer = torch.zeros(0, 5, device=app_state.device)
        app_state.buffer_size = [app_state.n_samples, 50000, 25 * app_state.n_samples, 25 * app_state.n_samples][app_state.algo]
        app_state.weights = None
        app_state.critic_loss = app_state.policy_loss = torch.tensor(float("nan"))
        torch.manual_seed(app_state.seed)
        # Head means plus one state-conditioned log std, which only SAC uses.
        app_state.actor = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, app_state.n_heads + 1)).to(app_state.device)
        app_state.gate = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, app_state.n_heads)).to(app_state.device)
        # V(s) for PPO/AWR/SCOOT, twin Q(s, a) for SAC.
        app_state.critic = nn.ModuleList([nn.Sequential(nn.Linear(1 + (app_state.algo == 1), 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)) for _ in range(1 + (app_state.algo == 1))]).to(app_state.device)
        if app_state.sobol_init:
            # Eq. 7 folded into the last layer: tiny state dependence, heads start at space-filling offsets.
            with torch.no_grad():
                app_state.actor[-1].weight[: app_state.n_heads] *= 1e-3
                app_state.actor[-1].bias[: app_state.n_heads] = 2 * torch.quasirandom.SobolEngine(1, scramble=True, seed=app_state.seed).draw(app_state.n_heads)[:, 0].to(app_state.device) - 1
        app_state.log_std = nn.Parameter(torch.full((1,), app_state.init_log_std, device=app_state.device))
        app_state.log_alpha = nn.Parameter(torch.zeros(1, device=app_state.device))
        app_state.policy_parameters = [*app_state.actor.parameters(), *app_state.gate.parameters(), app_state.log_std, app_state.log_alpha]
        app_state.optimizer = torch.optim.RAdam if app_state.algo >= 2 else torch.optim.Adam
        app_state.policy_optimizer = app_state.optimizer(app_state.policy_parameters, lr=app_state.policy_lr)
        app_state.critic_optimizer = app_state.optimizer(app_state.critic.parameters(), lr=app_state.critic_lr)
        app_state.refresh = True

    app_state.frame_stages = app_state.stages_per_frame if app_state.training else int(app_state.stage_clicked)
    for _ in range(app_state.frame_stages):
        app_state.learn_std = not app_state.curriculum or app_state.iteration >= app_state.std_start
        app_state.learn_gate = not app_state.curriculum or app_state.iteration >= app_state.gate_start
        if app_state.stage == 0:
            with torch.no_grad():
                app_state.s = 2 * torch.rand(app_state.n_samples, 1, generator=app_state.train_rng, device=app_state.device) - 1
                app_state.out = app_state.actor(app_state.s)
                app_state.std = (app_state.out[:, app_state.n_heads :].clamp(-20, 2) if app_state.algo == 1 else app_state.log_std).exp()
                app_state.log_p = app_state.gate(app_state.s).log_softmax(-1) if app_state.learn_gate else torch.full_like(app_state.out[:, : app_state.n_heads], -math.log(app_state.n_heads))
                app_state.head = torch.multinomial(app_state.log_p.exp(), 1, generator=app_state.train_rng)
                app_state.u = app_state.out.gather(1, app_state.head) + app_state.std * torch.randn(app_state.n_samples, 1, generator=app_state.train_rng, device=app_state.device)
                app_state.logp = torch.logsumexp(app_state.log_p + torch.distributions.Normal(app_state.out[:, : app_state.n_heads], app_state.std).log_prob(app_state.u), -1, keepdim=True)
                app_state.batch = torch.cat((app_state.s, app_state.u, app_state.u.tanh(), torch.zeros_like(app_state.s), app_state.logp), 1)

        elif app_state.stage == 1:
            with torch.no_grad():
                # Test action: mean of one head drawn from p_h(s), as in the paper.
                app_state.out = app_state.actor(app_state.test_s)
                app_state.log_p = app_state.gate(app_state.test_s).log_softmax(-1) if app_state.learn_gate else torch.full_like(app_state.out[:, : app_state.n_heads], -math.log(app_state.n_heads))
                app_state.test_a = app_state.out.gather(1, torch.multinomial(app_state.log_p.exp(), 1, generator=app_state.eval_rng)).tanh()
                app_state.query_s = torch.cat((app_state.batch[:, 0], app_state.test_s[:, 0]))
                app_state.query_a = torch.cat((app_state.batch[:, 2], app_state.test_a[:, 0]))
                # Exact 1-NN: the table is sorted by state, so the nearest shot lies in a small index window.
                index = (torch.searchsorted(app_state.table_s, app_state.query_s)[:, None] + app_state.window).clamp(0, len(app_state.table_s) - 1)
                distance = (app_state.table_s[index] - app_state.query_s[:, None]).square() + (app_state.table_a[index] - app_state.query_a[:, None]).square()
                app_state.query_r = app_state.table_r[index.gather(1, distance.argmin(1, keepdim=True))][:, 0]
                app_state.batch[:, 3] = app_state.query_r[: app_state.n_samples]
                app_state.test_r = app_state.query_r[app_state.n_samples :]
                app_state.buffer = torch.cat((app_state.buffer, app_state.batch))[-app_state.buffer_size :]
                app_state.weights = None
                app_state.curve = np.concatenate((app_state.curve, [[app_state.iteration * app_state.n_samples], [app_state.test_r.mean().item()], [(app_state.test_r > 0.5).float().mean().item()], [app_state.batch[:, 3].mean().item()]]), 1)

        elif app_state.stage == 2:
            # SAC: one step of 256 per sample (SB3 default). PPO: 10 epochs x 64. AWR/SCOOT: one epoch x 256.
            if app_state.algo == 1:
                app_state.minibatches = [torch.randint(len(app_state.buffer), (256,), generator=app_state.train_rng, device=app_state.device) for _ in range(app_state.n_samples)]
            else:
                app_state.minibatches = [index for _ in range(10 if app_state.algo == 0 else 1) for index in torch.randperm(len(app_state.buffer), generator=app_state.train_rng, device=app_state.device).split(64 if app_state.algo == 0 else 256)]
            for index in app_state.minibatches:
                app_state.minibatch = app_state.buffer[index]
                # Returns equal rewards (gamma = 0), so critics regress r directly without target networks.
                app_state.critic_loss = sum((critic(app_state.minibatch[:, [0, 2]] if app_state.algo == 1 else app_state.minibatch[:, :1]) - app_state.minibatch[:, 3:4]).square().mean() for critic in app_state.critic)
                app_state.critic_optimizer.zero_grad()
                app_state.critic_loss.backward()
                app_state.critic_optimizer.step()

        elif app_state.stage == 3 and app_state.algo != 1:
            with torch.no_grad():
                app_state.advantage = app_state.buffer[:, 3] - app_state.critic[0](app_state.buffer[:, :1])[:, 0]
                app_state.normalized = (app_state.advantage - app_state.advantage.mean()) / (app_state.advantage.std() + 1e-8)
                if app_state.algo == 0:
                    app_state.weights = app_state.normalized
                elif app_state.elite:
                    app_state.weights = app_state.advantage.clamp(min=0)
                else:
                    app_state.weights = (app_state.normalized / app_state.temperature).exp().clamp(max=20)

        elif app_state.stage == 4:
            if app_state.algo == 1:
                app_state.minibatches = [torch.randint(len(app_state.buffer), (256,), generator=app_state.train_rng, device=app_state.device) for _ in range(app_state.n_samples)]
            elif app_state.algo == 0:
                app_state.minibatches = [index for _ in range(10) for index in torch.randperm(len(app_state.buffer), generator=app_state.train_rng, device=app_state.device).split(64)]
            else:
                app_state.elites = app_state.weights.nonzero()[:, 0]
                app_state.minibatches = [app_state.elites[index] for index in torch.randperm(len(app_state.elites), generator=app_state.train_rng, device=app_state.device).split(256)]
            for index in app_state.minibatches:
                app_state.minibatch = app_state.buffer[index]
                app_state.out = app_state.actor(app_state.minibatch[:, :1])
                app_state.means = app_state.out[:, : app_state.n_heads]
                app_state.std = (app_state.out[:, app_state.n_heads :].clamp(-20, 2) if app_state.algo == 1 else app_state.log_std if app_state.learn_std else app_state.log_std.detach()).exp()
                if app_state.algo == 1:
                    app_state.u = app_state.means + app_state.std * torch.randn(len(index), 1, generator=app_state.train_rng, device=app_state.device)
                    app_state.logp = torch.distributions.Normal(app_state.means, app_state.std).log_prob(app_state.u) - (1 - app_state.u.tanh().square() + 1e-6).log()
                    app_state.q = torch.min(*(critic(torch.cat((app_state.minibatch[:, :1], app_state.u.tanh()), 1)) for critic in app_state.critic))
                    # Actor maximizes Q + entropy; temperature tracks target entropy -1 (= -action dims).
                    app_state.policy_loss = (app_state.log_alpha.exp().detach() * app_state.logp - app_state.q).mean() - (app_state.log_alpha * (app_state.logp.detach() - 1)).mean()
                else:
                    app_state.log_p = app_state.gate(app_state.minibatch[:, :1]).log_softmax(-1) if app_state.learn_gate else torch.full_like(app_state.means, -math.log(app_state.n_heads))
                    app_state.logp = torch.logsumexp(app_state.log_p + torch.distributions.Normal(app_state.means, app_state.std).log_prob(app_state.minibatch[:, 1:2]), -1)
                    if app_state.algo == 0:
                        app_state.ratio = (app_state.logp - app_state.minibatch[:, 4]).exp()
                        app_state.policy_loss = -torch.min(app_state.ratio * app_state.weights[index], app_state.ratio.clamp(0.8, 1.2) * app_state.weights[index]).mean()
                    else:
                        app_state.policy_loss = -(app_state.weights[index] * app_state.logp).mean()
                        if app_state.n_heads > 1:
                            # Eq. 6: penalize the closest pair of head means once they are within d stds.
                            app_state.gap = (app_state.means[:, :, None] - app_state.means[:, None, :]).abs().masked_fill(torch.eye(app_state.n_heads, dtype=torch.bool, device=app_state.device), float("inf")).amin((1, 2))
                            app_state.policy_loss = app_state.policy_loss + app_state.dist_weight * (1 - app_state.gap / (app_state.overlap * app_state.std)).clamp(min=0).mean()
                app_state.policy_optimizer.zero_grad()
                app_state.policy_loss.backward()
                if app_state.algo == 0:
                    nn.utils.clip_grad_norm_(app_state.policy_parameters, 0.5)
                app_state.policy_optimizer.step()

        app_state.stage = (app_state.stage + 1) % 5
        if app_state.stage == 0:
            app_state.iteration += 1
            if app_state.iteration >= app_state.max_iters:
                app_state.training = False
                break

    # Cache plot arrays only when something changed.
    if app_state.frame_stages or app_state.refresh:
        app_state.refresh = False
        app_state.learn_gate = not app_state.curriculum or app_state.iteration >= app_state.gate_start
        with torch.no_grad():
            app_state.shown = app_state.buffer[-3200:].cpu().numpy()
            app_state.shown_s = app_state.shown[:, 0].copy()
            app_state.shown_a = app_state.shown[:, 2].copy()
            # ImPlot bindings reject empty arrays, so empty buffers and curves are simply not drawn.
            if len(app_state.shown):
                app_state.buffer_style.marker_fill_colors = np.ascontiguousarray((app_state.blues(app_state.shown[:, 3].clip(0, 1)) * 255).astype(np.uint8)).view(np.uint32).ravel()
                app_state.shown_w = app_state.weights[-3200:].cpu().numpy() if app_state.weights is not None else np.ones(len(app_state.shown))
                app_state.buffer_style.marker_sizes = (1.5 + 3.5 * (app_state.shown_w - app_state.shown_w.min()) / (np.ptp(app_state.shown_w) + 1e-12) if app_state.weights is not None else np.full(len(app_state.shown), 3.0)).astype(np.float32)
            if app_state.stage == 1:
                app_state.batch_plot = app_state.batch.cpu().numpy().T.copy()
            app_state.grid_out = app_state.actor(app_state.grid_s)
            app_state.grid_p = app_state.gate(app_state.grid_s).softmax(-1) if app_state.learn_gate else torch.full_like(app_state.grid_out[:, : app_state.n_heads], 1 / app_state.n_heads)
            app_state.means_x = app_state.grid_s[:, 0].repeat(app_state.n_heads).cpu().numpy()
            app_state.means_y = app_state.grid_out[:, : app_state.n_heads].tanh().T.flatten().cpu().numpy()
            app_state.mean_style.marker_line_colors = np.uint32(0x0080FF) | (255 * (0.2 + 0.8 * app_state.grid_p.T.flatten().cpu().numpy())).astype(np.uint32) << 24
            app_state.std_value = (app_state.grid_out[:, app_state.n_heads].clamp(-20, 2).exp().mean() if app_state.algo == 1 else app_state.log_std.exp()).item()

    imgui.separator()
    imgui.text(f"{app_state.device} | iteration {app_state.iteration} | samples {app_state.iteration * app_state.n_samples} | buffer {len(app_state.buffer)}/{app_state.buffer_size} | next: {app_state.stage_names[app_state.stage]}")
    if app_state.algo >= 2 and app_state.curriculum:
        imgui.text(f"Curriculum stage {1 + (app_state.iteration >= app_state.std_start) + (app_state.iteration >= app_state.gate_start)}: learning means" + ", std" * (app_state.iteration >= app_state.std_start) + ", gating" * (app_state.iteration >= app_state.gate_start))
    imgui.text(f"Policy std: {app_state.std_value:.4f}" + (f" | entropy temperature: {app_state.log_alpha.exp().item():.4f}" if app_state.algo == 1 else ""))
    imgui.text(f"Critic loss: {app_state.critic_loss.item():.6f} | policy loss: {app_state.policy_loss.item():.6f}")
    if app_state.algo >= 2 and app_state.weights is not None:
        imgui.text(f"Samples with nonzero weight: {(app_state.weights > 0).float().mean().item():.3f}")
    if app_state.curve.shape[1]:
        imgui.text(f"Test return: {app_state.curve[1, -1]:.3f} (best {app_state.curve[1].max():.3f}) | test success rate: {app_state.curve[2, -1]:.3f}")
    if implot.begin_plot("Learning curve", (-1, max(250, imgui.get_content_region_avail().y))):
        implot.setup_axes("Samples", "Return", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
        if app_state.curve.shape[1]:
            implot.plot_line("Test return (2048 fixed states, head means)", app_state.curve[0], app_state.curve[1])
            implot.plot_line("Batch mean reward (sampled actions)", app_state.curve[0], app_state.curve[3])
        if app_state.algo >= 2 and app_state.curriculum:
            implot.plot_inf_lines("Curriculum stages", np.array([app_state.std_start, app_state.gate_start], dtype=np.float64) * app_state.n_samples, app_state.guide_style)
        implot.end_plot()
    imgui.pop_item_width()
    imgui.end_child()

    imgui.same_line()
    imgui.begin_child("Landscape", (0, 0))
    if implot.begin_plot("Reward landscape", (-1, -1), implot.Flags_.equal):
        implot.setup_axes("State (cue ball placement)", "Action (shot angle)")
        implot.setup_axes_limits(-1, 1, -1, 1)
        implot.plot_image("Reward landscape", imgui.ImTextureRef(hello_imgui.im_texture_id_from_asset("1s1a_landscape.png")), implot.Point(-1, -1), implot.Point(1, 1))
        if len(app_state.shown):
            implot.plot_scatter("Replay buffer (color: reward, size: weight)", app_state.shown_s, app_state.shown_a, app_state.buffer_style)
        if app_state.stage == 1:
            implot.plot_scatter("New samples (not yet evaluated)", app_state.batch_plot[0], app_state.batch_plot[2], app_state.batch_style)
        implot.plot_scatter("Policy means (opacity: p_h(s))", app_state.means_x, app_state.means_y, app_state.mean_style)
        implot.end_plot()
    imgui.end_child()


class AppState:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(Path(__file__).with_name("1s1a_precomputed.pth"), "rb") as file:
            data = pickle.load(file)
        table = torch.as_tensor(np.concatenate((data["obses"], data["params"], data["rewards"]), 1), dtype=torch.float32, device=self.device)
        self.table_s, self.table_a, self.table_r = table[table[:, 0].argsort()].T.contiguous()
        # 8.7M uniform shots: +-8192 neighbours by state cover about +-0.0019, far beyond the typical nearest distance (~0.0003).
        self.window = torch.arange(-8192, 8192, device=self.device)
        self.algos = ["PPO", "SAC", "AWR", "SCOOT"]
        self.presets = [
            dict(policy_lr=3e-4, critic_lr=3e-4, n_heads=1, elite=False, sobol_init=False, dist_weight=0.0, curriculum=False),
            dict(policy_lr=3e-4, critic_lr=3e-4, n_heads=1, elite=False, sobol_init=False, dist_weight=0.0, curriculum=False),
            dict(policy_lr=1e-3, critic_lr=1e-5, n_heads=1, elite=False, sobol_init=False, dist_weight=0.0, curriculum=False),
            dict(policy_lr=1e-3, critic_lr=1e-5, n_heads=4, elite=True, sobol_init=True, dist_weight=0.1, curriculum=True),
        ]
        self.descriptions = [
            "PPO (Schulman et al. 2017): on-policy clipped surrogate (0.2) on the current batch; advantage r - V(s), normalized; learned global std; Adam, 10 epochs x 64, grad clip 0.5. The critic is fit on the batch before advantages are computed.",
            "SAC (Haarnoja et al. 2018): off-policy, 50k replay; state-conditioned std; twin Q regress r; actor maximizes min Q + entropy with automatic temperature (target -1); Adam, one gradient step of 256 per sample (slow).",
            "AWR (Peng et al. 2019): off-policy weighted regression on a 25-iteration FIFO buffer; weights exp(A / beta) with normalized A, clipped at 20; value lr 1e-5; learned global std; RAdam, one epoch x 256.",
            "SCOOT (Kim et al. 2022): AWR + elite samples (A > 0, weight = A), H-head Gaussian mixture with non-overlapping init and distance regularization (lambda 0.1, d = 1 std), and a 3-stage curriculum.",
        ]
        self.algo = 3
        self.__dict__.update(self.presets[self.algo])
        self.seed = 0
        self.n_samples = 128
        self.init_log_std = -1.0
        self.temperature = 1.0
        self.overlap = 1.0
        self.std_start = 500
        self.gate_start = 3000
        self.max_iters = 3500
        self.stages_per_frame = 5
        self.stage_names = ["Sample", "Evaluate", "Fit critic", "Weigh", "Learn"]
        self.stage = 0
        self.training = False
        self.reset = True
        self.blues = colormaps["Blues"]
        self.buffer_style = implot.Spec(marker=implot.Marker_.circle, marker_line_color=(0.35, 0.35, 0.35, 1), line_weight=0.5)
        self.batch_style = implot.Spec(marker=implot.Marker_.circle, marker_size=4, marker_fill_color=(0, 0, 0, 0), marker_line_color=(0, 0, 0, 1))
        self.mean_style = implot.Spec(marker=implot.Marker_.cross, marker_size=5, line_weight=2)
        self.guide_style = implot.Spec(line_color=(0.6, 0.6, 0.6, 1), flags=implot.ItemFlags_.no_fit)


hello_imgui.set_assets_folder(str(Path(__file__).parent))
app_state = AppState()
immapp.run(partial(gui, app_state), with_implot=True, fps_idle=0, ini_disable=True, window_title="SCOOT", window_size=(1700, 950))
