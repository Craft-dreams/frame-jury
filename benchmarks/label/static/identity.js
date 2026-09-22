let currentState = null;
let selectedFaces = new Set();

const byId = (id) => document.getElementById(id);

function setMessage(message, error = false) {
  const node = byId("identity-message");
  node.textContent = message;
  node.classList.toggle("error", error);
}

function updateSelection() {
  document.querySelectorAll("[data-face-index]").forEach((button) => {
    const selected = selectedFaces.has(Number(button.dataset.faceIndex));
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function toggleFace(index) {
  if (selectedFaces.has(index)) selectedFaces.delete(index);
  else selectedFaces.add(index);
  updateSelection();
  setMessage("");
}

function drawFaceBoxes() {
  if (!currentState?.case) return;
  const image = byId("identity-frame");
  if (!image.naturalWidth || !image.naturalHeight) return;
  const scaleX = image.clientWidth / image.naturalWidth;
  const scaleY = image.clientHeight / image.naturalHeight;
  const boxes = currentState.frame_faces.map((face, index) => {
    const [x0, y0, x1, y1] = face.box;
    const box = document.createElement("div");
    box.className = "face-box";
    box.textContent = String(index + 1);
    box.style.left = `${x0 * scaleX}px`;
    box.style.top = `${y0 * scaleY}px`;
    box.style.width = `${(x1 - x0) * scaleX}px`;
    box.style.height = `${(y1 - y0) * scaleY}px`;
    return box;
  });
  byId("face-boxes").replaceChildren(...boxes);
}

function renderFace(face, index) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "identity-face";
  button.dataset.faceIndex = String(face.index);
  button.setAttribute("aria-pressed", "false");
  const number = document.createElement("strong");
  number.textContent = String(index + 1);
  const image = document.createElement("img");
  image.src = face.url;
  image.alt = `Rosto ${index + 1} do quadro`;
  button.append(number, image);
  button.addEventListener("click", () => toggleFace(face.index));
  return button;
}

async function loadNext() {
  const response = await fetch("/api/identity/next", {cache: "no-store"});
  const state = await response.json();
  if (!response.ok) throw new Error(state.error || "Não foi possível carregar o próximo caso");
  currentState = state;
  selectedFaces.clear();
  setMessage("");
  byId("identity-progress").textContent = `${state.labelled} de ${state.total}`;
  if (!state.case) {
    byId("identity-workspace").hidden = true;
    byId("identity-done").hidden = false;
    return;
  }

  byId("identity-workspace").hidden = false;
  byId("identity-done").hidden = true;
  byId("identity-name").textContent = state.entity.display_name;
  byId("visual-identity").textContent = state.entity.visual_identity || "Identidade visual não informada";
  const referenceUrl = state.entity.reference_urls[0];
  const hasReferenceFace = state.reference_faces.length > 0;
  const referenceFace = byId("reference-face");
  referenceFace.src = hasReferenceFace ? state.reference_faces[0].url : referenceUrl;
  referenceFace.classList.toggle("full-reference-large", !hasReferenceFace);
  byId("no-reference-face").hidden = hasReferenceFace;
  const referenceFull = byId("reference-full");
  referenceFull.src = referenceUrl;
  referenceFull.hidden = !hasReferenceFace;

  byId("frame-faces").replaceChildren(...state.frame_faces.map(renderFace));
  const noFaces = state.frame_faces.length === 0;
  byId("no-frame-faces").hidden = !noFaces;
  byId("same").disabled = noFaces;
  const frame = byId("identity-frame");
  frame.src = state.case.image_url;
  byId("face-boxes").replaceChildren();
}

async function submit(decision) {
  if (!currentState?.case) return;
  if (decision === "same" && byId("same").disabled) return;
  let faces = [...selectedFaces].sort((a, b) => a - b);
  if (decision === "same" && !faces.length) {
    setMessage("Selecione pelo menos um rosto", true);
    return;
  }
  if (decision === "not_visible" || decision === "unsure") faces = [];
  const response = await fetch("/api/identity/labels", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      case_id: currentState.case.case_id,
      entity_id: currentState.entity_id,
      decision,
      faces,
    }),
  });
  const result = await response.json();
  if (!response.ok) {
    setMessage(result.error || "Não foi possível salvar a decisão", true);
    return;
  }
  await loadNext();
}

byId("same").addEventListener("click", () => submit("same"));
byId("different").addEventListener("click", () => submit("different"));
byId("not-visible").addEventListener("click", () => submit("not_visible"));
byId("unsure").addEventListener("click", () => submit("unsure"));
byId("identity-frame").addEventListener("load", drawFaceBoxes);
window.addEventListener("resize", drawFaceBoxes);

document.addEventListener("keydown", (event) => {
  if (event.key >= "1" && event.key <= "9") {
    const index = Number(event.key) - 1;
    if (index < (currentState?.frame_faces.length || 0)) toggleFace(index);
  } else if (event.key === "Enter") submit("same");
  else if (event.key.toLowerCase() === "d") submit("different");
  else if (event.key.toLowerCase() === "a") submit("not_visible");
  else if (event.key.toLowerCase() === "u") submit("unsure");
  else if (event.key === "Escape") {
    selectedFaces.clear();
    updateSelection();
    setMessage("");
  }
});

loadNext().catch((error) => setMessage(error.message, true));
