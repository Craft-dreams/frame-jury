const shortcuts = ["1", "2", "3", "4", "5", "6", "7", "8"];
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
  const heading = document.createElement("h3");
  heading.textContent = `${entity.entity_id} · ${entity.kind}`;
  wrapper.appendChild(heading);
  const references = document.createElement("div");
  references.className = "references";
  for (const [index, url] of entity.reference_urls.entries()) {
    const image = document.createElement("img");
    image.src = url;
    image.alt = `Reference ${index + 1} for ${entity.entity_id}`;
    references.appendChild(image);
  }
  if (!entity.reference_urls.length) {
    references.textContent = "No reference image";
    references.classList.add("empty");
  }
  wrapper.appendChild(references);
  return wrapper;
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
  byId("progress").textContent = `${state.labelled} / ${state.total} labelled · ${state.labeller}`;
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
  byId("staging").textContent = currentCase.shot.staging;
  byId("positive-prompt").textContent = currentCase.shot.positive_prompt;
  byId("negative-prompt").textContent = currentCase.shot.negative_prompt || "None";
  const entities = byId("entities");
  entities.replaceChildren(...currentCase.shot.declared_entities.map(renderEntity));
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
  state.defects.forEach((defect, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.defect = defect;
    button.dataset.shortcut = shortcuts[index];
    button.innerHTML = `<kbd>${shortcuts[index]}</kbd> ${defect.replaceAll("_", " ")}`;
    button.addEventListener("click", () => toggleDefect(defect, button));
    container.appendChild(button);
  });
  document.querySelectorAll("[data-exclusive]").forEach((button) => {
    button.addEventListener("click", () => submit([button.dataset.exclusive]));
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
  const defectButton = document.querySelector(`[data-shortcut="${key}"]`);
  if (defectButton) defectButton.click();
  else if (key === "c") submit(["clean"]);
  else if (key === "u") submit(["uncertain"]);
  else if (key === "n") byId("notes").focus();
  else if (event.key === "Enter") submit([...selected]);
});

initialise().catch((error) => setMessage(error.message, true));
