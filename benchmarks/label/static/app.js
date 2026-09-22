const shortcuts = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="];
let currentCase = null;
let selected = new Set();

const byId = (id) => document.getElementById(id);

function setMessage(message, error = false) {
  const node = byId("message");
  node.textContent = message;
  node.classList.toggle("error", error);
}

function renderEntity(entity) {
  const wrapper = document.createElement("article");
  wrapper.className = "entity";
  const references = document.createElement("div");
  references.className = "references";
  for (const [index, url] of entity.reference_urls.entries()) {
    const image = document.createElement("img");
    image.src = url;
    image.alt = `Reference ${index + 1} for ${entity.display_name}`;
    references.appendChild(image);
  }
  if (!entity.reference_urls.length) {
    references.textContent = "No reference image";
    references.classList.add("empty");
  }
  wrapper.appendChild(references);
  const description = document.createElement("div");
  description.className = "entity-description";
  const heading = document.createElement("h3");
  heading.textContent = `${entity.display_name} (${entity.kind})`;
  description.appendChild(heading);
  const identity = document.createElement("p");
  identity.className = "visual-identity";
  identity.textContent = entity.visual_identity || "Visual profile unavailable";
  identity.title = identity.textContent;
  description.appendChild(identity);
  if (entity.relative_scale || entity.approximate_dimensions) {
    const scale = document.createElement("p");
    scale.className = "scale";
    scale.textContent = `Scale: ${[entity.relative_scale, entity.approximate_dimensions].filter(Boolean).join(" · ")}`;
    description.appendChild(scale);
  }
  const trace = document.createElement("small");
  const aliases = entity.aliases.length ? ` · aliases: ${entity.aliases.join(", ")}` : "";
  trace.textContent = `${entity.entity_id}${aliases}`;
  description.appendChild(trace);
  wrapper.appendChild(description);
  return wrapper;
}

function renderProgress(state) {
  byId("progress-summary").textContent = `${state.labelled} / ${state.total} labelled · ${state.labeller}`;
  const counts = Object.entries(state.positive_counts)
    .map(([defect, count]) => `${defect} ${count}`)
    .join(" · ");
  const legacy = state.legacy_broken_anatomy
    ? ` · broken_anatomy (legacy, unspecified) ${state.legacy_broken_anatomy}`
    : "";
  byId("positive-counts").textContent = `${counts}${legacy}`;
}

function toggleDefect(defect, button) {
  if (selected.has(defect)) selected.delete(defect);
  else selected.add(defect);
  button.classList.toggle("selected", selected.has(defect));
  setMessage("");
}

async function loadNext() {
  const response = await fetch("/api/next", {cache: "no-store"});
  const state = await response.json();
  renderProgress(state);
  currentCase = state.case;
  selected.clear();
  document.querySelectorAll("[data-defect]").forEach((button) => button.classList.remove("selected"));
  byId("notes").value = "";
  setMessage("");
  if (!currentCase) {
    byId("workspace").hidden = true;
    byId("done").hidden = false;
    return;
  }
  byId("workspace").hidden = false;
  byId("done").hidden = true;
  byId("frame").src = currentCase.image_url;
  byId("case-id").textContent = currentCase.case_id;
  byId("framing").textContent = currentCase.shot.framing;
  byId("purpose").textContent = currentCase.shot.staging.purpose;
  byId("must-render").textContent = currentCase.shot.staging.must_render.join(" · ") || "None declared";
  byId("composition").textContent = currentCase.shot.staging.composition.join(" · ") || "None declared";
  byId("positive-prompt").textContent = currentCase.shot.positive_prompt;
  byId("negative-prompt").textContent = currentCase.shot.negative_prompt || "None";
  const entities = byId("entities");
  entities.replaceChildren(...currentCase.shot.declared_entities.map(renderEntity));
  byId("no-character").hidden = currentCase.shot.declared_entities.some(
    (entity) => entity.kind === "character",
  );
}

async function submit(defects) {
  if (!currentCase) return;
  if (!defects.length) {
    setMessage("Select at least one defect, or choose clean/uncertain.", true);
    return;
  }
  const response = await fetch("/api/labels", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({case_id: currentCase.case_id, defects, notes: byId("notes").value}),
  });
  const result = await response.json();
  if (!response.ok) {
    setMessage(result.error || "Could not save label.", true);
    return;
  }
  await loadNext();
}

async function initialise() {
  const state = await (await fetch("/api/next", {cache: "no-store"})).json();
  const container = byId("defect-buttons");
  const descriptions = state.descriptions || {};
  state.defects.forEach((defect, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.defect = defect;
    button.dataset.shortcut = shortcuts[index];
    const desc = descriptions[defect] || "";
    const descHtml = desc ? `<span class="defect-desc">${desc}</span>` : "";
    button.innerHTML = `<span class="defect-name"><kbd>${shortcuts[index]}</kbd> ${defect.replaceAll("_", " ")}</span>${descHtml}`;
    button.addEventListener("click", () => toggleDefect(defect, button));
    container.appendChild(button);
  });
  document.querySelectorAll("[data-exclusive]").forEach((button) => {
    const key = button.dataset.exclusive;
    const desc = descriptions[key] || "";
    const kbd = key === "clean" ? "C" : "U";
    const descHtml = desc ? `<span class="defect-desc">${desc}</span>` : "";
    button.innerHTML = `<span class="defect-name"><kbd>${kbd}</kbd> ${key}</span>${descHtml}`;
    button.addEventListener("click", () => submit([key]));
  });
  byId("save").addEventListener("click", () => submit([...selected]));
  await loadNext();
}

document.addEventListener("keydown", (event) => {
  if (event.target.matches("textarea, input")) {
    if (event.key === "Escape") event.target.blur();
    return;
  }
  const key = event.key.toLowerCase();
  if (shortcuts.includes(key)) {
    const defectButton = document.querySelector(`[data-shortcut="${key}"]`);
    if (defectButton) defectButton.click();
  } else if (key === "c") submit(["clean"]);
  else if (key === "u") submit(["uncertain"]);
  else if (key === "n") byId("notes").focus();
  else if (event.key === "Enter") submit([...selected]);
});

initialise().catch((error) => setMessage(error.message, true));
