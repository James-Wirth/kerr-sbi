const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";
const count = (value) =>
  Number.isFinite(value) ? value.toLocaleString() : "—";
const bytes = (value) =>
  value >= 2 ** 30
    ? `${(value / 2 ** 30).toFixed(1)} GiB`
    : `${(value / 2 ** 20).toFixed(1)} MiB`;
const duration = (value) =>
  !Number.isFinite(value)
    ? "—"
    : value < 60
      ? `${value.toFixed(1)} s`
      : value < 3600
        ? `${Math.floor(value / 60)}m ${Math.floor(value % 60)}s`
        : `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`;
const statuses = {
  complete: "Completed",
  early_stopping: "Early stopping",
  overfit_target: "Overfit target met",
  time_limit: "Time limit",
  paused: "Paused",
  interrupted: "Interrupted",
  failed: "Failed",
  running: "Training",
  initializing: "Preparing",
  evaluating: "Evaluating",
  saving: "Saving checkpoint",
  unconfirmed: "No recent update",
  unavailable: "No live record",
};
let selected = new URLSearchParams(location.search).get("run"),
  overview,
  currentRun,
  polling = false,
  previewToken = 0;
async function api(path) {
  const response = await fetch(path, { cache: "no-store" });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Cannot read local records");
  return value;
}
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function pairs(target, rows) {
  $(target).replaceChildren(
    ...rows.map(([key, value]) => {
      const row = el("div");
      row.append(el("dt", key), el("dd", value));
      return row;
    }),
  );
}
function renderRuns() {
  const filter = $("run-filter").value;
  const runs = overview.runs.filter(
    (run) => filter === "all" || run.dummy === (filter === "dummy"),
  );
  $("run-list").replaceChildren(
    ...runs.map((run) => {
      const button = el(
        "button",
        undefined,
        `run-button${selected === run.id ? " active" : ""}`,
      );
      button.setAttribute(
        "aria-current",
        selected === run.id ? "true" : "false",
      );
      button.append(
        el("strong", run.name),
        el(
          "small",
          `${run.dummy ? "Dummy" : "Physical"} · ${statuses[run.status] || run.status}`,
        ),
      );
      button.onclick = () => {
        selected = run.id;
        const url = new URL(location);
        url.searchParams.set("run", selected);
        history.replaceState(null, "", url);
        renderRuns();
        refreshRun();
      };
      return button;
    }),
  );
  if (!runs.length)
    $("run-list").append(
      el("p", "No experiments in this group.", "muted small"),
    );
}
function chart(history) {
  const root = $("loss-chart");
  root.replaceChildren();
  if (!history.length) {
    root.append(el("div", "Waiting for the first evaluation.", "empty"));
    return;
  }
  const ns = "http://www.w3.org/2000/svg",
    width = root.clientWidth || 980,
    height = root.clientHeight || 300,
    left = 62,
    right = 20,
    top = 18,
    bottom = 40;
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    "Training and validation negative log likelihood against optimizer step",
  );
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  };
  const series = [
    ["train_nll", "#155ddb"],
    ["validation_nll", "#007d74"],
    ["prior_validation_nll", "#929da5"],
  ];
  const values = history
    .flatMap((row) => series.map(([key]) => row[key]))
    .filter(Number.isFinite);
  if (!values.length) {
    root.append(el("div", "No finite evaluations available.", "empty"));
    return;
  }
  let lo = Math.min(...values),
    hi = Math.max(...values),
    padding = Math.max((hi - lo) * 0.16, 0.08);
  lo -= padding;
  hi += padding;
  const maxStep = Math.max(history.at(-1).step, 1),
    x = (value) => left + (value / maxStep) * (width - left - right),
    y = (value) =>
      height - bottom - ((value - lo) / (hi - lo)) * (height - top - bottom);
  for (let i = 0; i <= 4; i++) {
    const value = lo + ((hi - lo) * i) / 4;
    add("line", {
      x1: left,
      y1: y(value),
      x2: width - right,
      y2: y(value),
      stroke: "#e8eef2",
    });
    add(
      "text",
      { x: left - 12, y: y(value) + 4, "text-anchor": "end" },
      value.toFixed(2),
    );
    const step = (maxStep * i) / 4;
    add(
      "text",
      { x: x(step), y: height - 16, "text-anchor": "middle" },
      Math.round(step).toLocaleString(),
    );
  }
  series.forEach(([key, color]) => {
    const points = history.filter((r) => Number.isFinite(r[key]));
    add("polyline", {
      points: points.map((r) => `${x(r.step)},${y(r[key])}`).join(" "),
      fill: "none",
      stroke: color,
      "stroke-width": key.startsWith("prior") ? 1.5 : 2.5,
      "stroke-dasharray": key.startsWith("prior") ? "6 5" : "none",
      "stroke-linejoin": "round",
    });
    points.forEach((row) => {
      const circle = add("circle", {
        cx: x(row.step),
        cy: y(row[key]),
        r: 3,
        fill: color,
      });
      const title = document.createElementNS(ns, "title");
      title.textContent = `Step ${row.step}: ${key} ${row[key].toFixed(4)}`;
      circle.append(title);
    });
  });
  root.append(svg);
}
function renderRun(run) {
  currentRun = run;
  $("run-kind").textContent = run.dummy
    ? "DUMMY EXPERIMENT"
    : "PHYSICAL IMAGE EXPERIMENT";
  $("run-title").textContent = run.name;
  $("run-subtitle").textContent =
    `${run.device} · ${run.settings.flow_layers}-layer spline flow · σ noise ${run.noise} · seed ${run.settings.seed}`;
  $("run-status").textContent = statuses[run.status] || run.status;
  $("run-status").className =
    `status${["running", "initializing", "evaluating", "saving"].includes(run.status) ? " live" : ""}`;
  let caution = run.dummy
    ? "Dummy data checks software behavior. These results do not establish Kerr inference or calibration."
    : run.train_count < 256
      ? "Small physical smoke test. This dataset is sufficient to exercise training, but not to assess generalization or calibration."
      : "";
  if (run.status === "unconfirmed")
    caution +=
      " No event has arrived for 30 seconds. The process may be compiling, evaluating or no longer running.";
  if (run.error) caution += ` ${run.error}`;
  $("run-caution").textContent = caution;
  $("run-caution").hidden = !caution;
  $("step-value").textContent =
    `${count(run.step)} / ${count(run.total_steps)}`;
  $("step-bar").max = Math.max(run.total_steps, 1);
  $("step-bar").value = run.step;
  $("epoch-value").textContent =
    `Epoch ${fmt(run.epoch, 1)} · ${run.settings.batch_size} images per batch`;
  $("best-value").textContent = fmt(run.best_validation);
  $("prior-value").textContent =
    `Validation prior ${fmt(run.current.prior_validation_nll)}`;
  $("elapsed-value").textContent = duration(run.elapsed);
  const updateTime = run.last_update.update_seconds;
  $("speed-value").textContent = updateTime
    ? `${fmt(updateTime * 1000, 0)} ms / latest update`
    : "Archived run · step timings unavailable";
  if (Number.isFinite(run.remaining_seconds))
    $("speed-value").textContent +=
      ` · ≈ ${duration(run.remaining_seconds)} to step limit`;
  $("rows-value").textContent = `${count(run.train_count)} images`;
  const hasTest = overview.datasets.some(
    (d) => d.split === "test" && d.dummy === run.dummy && d.count > 0,
  );
  $("validation-value").textContent =
    `${count(run.validation_count)} validation · ${hasTest ? "test held out" : "test not generated"}`;
  chart(run.history);
  $("loss-caption").textContent = run.events_available
    ? "Training and validation use fixed evaluation noise. Values update after each evaluation; optimizer progress updates during batches."
    : "Archived checkpoint evaluations. Per-batch telemetry was not recorded for this run.";
  $("loss-caption").textContent +=
    ` Training prior NLL: ${fmt(run.current.prior_train_nll)}.`;
  pairs("run-config", [
    ["Batch size", count(run.settings.batch_size)],
    ["Peak learning rate", run.settings.learning_rate?.toExponential(1) || "—"],
    ["Warmup", `${count(run.settings.warmup_steps)} updates`],
    ["Early-stopping patience", `${count(run.settings.patience)} evaluations`],
    ["Gradient norm (before clipping)", fmt(run.last_update.gradient_norm)],
    ["Latest batch NLL", fmt(run.last_update.batch_nll)],
    ["Asinh scale", fmt(run.normalization.asinh_scale)],
    ["JAX", run.versions.jax || "—"],
  ]);
  $("checkpoint-info").textContent = run.checkpoint
    ? `Published at step ${run.checkpoint_step}. ${run.step > run.checkpoint_step ? `${run.step - run.checkpoint_step} later updates are not checkpointed.` : "Best weights and optimizer state saved."}`
    : "No checkpoint published yet. This trainer saves at a requested pause, graceful interruption or completion.";
  $("evaluations").replaceChildren(
    ...run.history
      .slice(-8)
      .reverse()
      .map((row) => {
        const tr = el("tr");
        [
          count(row.step),
          fmt(row.train_nll, 4),
          fmt(row.validation_nll, 4),
        ].forEach((value) => tr.append(el("td", value)));
        return tr;
      }),
  );
}
function renderOverview() {
  const storage = overview.storage;
  $("disk-free").textContent = `${bytes(storage.free)} free`;
  $("disk-detail").textContent =
    `${bytes(storage.data + storage.runs + storage.results)} in data, runs and results`;
  $("planned-size").textContent = bytes(storage.planned_dataset);
  $("storage-breakdown").replaceChildren(
    ...[
      ["Data", storage.data],
      ["Checkpoints & runs", storage.runs],
      ["Results", storage.results],
    ].map(([name, size]) => {
      const row = el("div", undefined, "storage-row");
      row.append(el("span", name), el("span", bytes(size)));
      return row;
    }),
  );
  if (!selected) selected = overview.runs[0]?.id;
  renderRuns();
  const previous = $("dataset-select").value;
  const datasets = overview.datasets.filter((d) => d.split === "train");
  $("dataset-select").replaceChildren(
    ...datasets.map((d) => {
      const option = el("option", `${d.id} (${count(d.count)})`);
      option.value = d.id;
      return option;
    }),
  );
  if (datasets.some((d) => d.id === previous))
    $("dataset-select").value = previous;
  else if (datasets.some((d) => d.id === "train"))
    $("dataset-select").value = "train";
  $("dataset-inventory").replaceChildren(
    ...overview.datasets.map((d) => {
      const row = el("div", undefined, "dataset-row");
      row.append(
        el("span", `${d.id}${d.dummy ? " · dummy" : ""}`),
        el(
          "span",
          `${count(d.count)} / ${count(d.requested)} images${d.failed ? ` · ${d.failed} failed` : ""}`,
        ),
      );
      return row;
    }),
  );
  if (!datasets.length)
    $("dataset-inventory").append(
      el("p", "No preprocessed training snapshots found.", "muted small"),
    );
  $("notice").hidden = !overview.unreadable_runs.length;
  $("notice").textContent =
    `Could not read ${overview.unreadable_runs.length} run record(s). Other runs are still available.`;
}
async function refreshRun() {
  const requested = selected;
  if (requested) {
    const run = await api(`/api/run?id=${encodeURIComponent(requested)}`);
    if (selected === requested) renderRun(run);
  }
}
async function preview() {
  const token = ++previewToken,
    dataset = overview?.datasets.find(
      (d) => d.id === $("dataset-select").value,
    );
  if (!dataset) return;
  const index = Math.max(
    0,
    Math.min(dataset.count - 1, Number($("image-index").value) || 0),
  );
  $("image-index").value = index;
  $("image-index").max = dataset.count - 1;
  $("previous").disabled = index === 0;
  $("next").disabled = index >= dataset.count - 1;
  try {
    const data = await api(
      `/api/preview?id=${encodeURIComponent(dataset.id)}&index=${index}`,
    );
    if (token !== previewToken) return;
    const ctx = $("preview").getContext("2d"),
      image = ctx.createImageData(64, 64);
    data.pixels.forEach((value, i) => {
      image.data[i * 4] = Math.round(value * 0.92);
      image.data[i * 4 + 1] = Math.round(value * 0.97);
      image.data[i * 4 + 2] = value;
      image.data[i * 4 + 3] = 255;
    });
    ctx.putImageData(image, 0, 0);
    $("image-label").textContent =
      `Image ${data.idx.toString().padStart(6, "0")}`;
    pairs("image-stats", [
      ["Spin a", fmt(data.spin, 3)],
      ["Inclination", `${fmt(data.inclination, 1)}°`],
      ["Peak", fmt(data.peak, 1)],
      ["Flux", fmt(data.flux, 0)],
    ]);
  } catch (error) {
    if (token !== previewToken) return;
    $("image-label").textContent = error.message;
    $("preview").getContext("2d").clearRect(0, 0, 64, 64);
    $("image-stats").replaceChildren();
  }
}
async function refresh() {
  if (polling) return;
  polling = true;
  try {
    overview = await api("/api/overview");
    renderOverview();
    await refreshRun();
    $("connection").textContent = "Local connection";
    $("connection-dot").className = "dot online";
    $("last-refresh").textContent =
      `Updated ${new Date().toLocaleTimeString()}`;
    if (!$("preview").dataset.loaded) {
      await preview();
      $("preview").dataset.loaded = "yes";
    }
  } catch (error) {
    $("connection").textContent = "Disconnected";
    $("connection-dot").className = "dot";
    $("notice").hidden = false;
    $("notice").textContent =
      `${error.message}. Retrying automatically; last readings remain visible.`;
  } finally {
    polling = false;
  }
}
$("refresh").onclick = refresh;
$("run-filter").onchange = renderRuns;
$("dataset-select").onchange = () => {
  $("image-index").value = 0;
  preview();
};
$("image-index").onchange = preview;
$("previous").onclick = () => {
  $("image-index").value = Number($("image-index").value) - 1;
  preview();
};
$("next").onclick = () => {
  $("image-index").value = Number($("image-index").value) + 1;
  preview();
};
refresh();
setInterval(() => {
  if (!document.hidden) refresh();
}, 2000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
new ResizeObserver(() => {
  if (currentRun) chart(currentRun.history);
}).observe($("loss-chart"));
