class BrainNetworkVisual extends HTMLElement {
  constructor() {
    super();
    this.canvas = document.createElement("canvas");
    this.context = this.canvas.getContext("2d", { alpha: true });
    this.pointer = { x: 0, y: 0 };
    this.geometry = null;
    this.width = 0;
    this.height = 0;
    this.pixelRatio = 1;
    this.prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.frameRequest = null;
    this.palette = [
      [67, 220, 190],
      [250, 128, 105],
      [242, 192, 90],
      [125, 174, 255],
    ];

    this.resize = this.resize.bind(this);
    this.render = this.render.bind(this);
    this.updatePointer = this.updatePointer.bind(this);
  }

  connectedCallback() {
    this.classList.add("brain-network-visual");
    this.canvas.setAttribute("aria-hidden", "true");
    this.append(this.canvas);
    this.readSettings();
    this.loadGeometry().then(() => {
      this.resize();
      window.addEventListener("resize", this.resize, { passive: true });
      window.addEventListener("pointermove", this.updatePointer, { passive: true });
      this.render(performance.now());
    });
  }

  disconnectedCallback() {
    window.removeEventListener("resize", this.resize);
    window.removeEventListener("pointermove", this.updatePointer);
    if (this.frameRequest) {
      cancelAnimationFrame(this.frameRequest);
    }
  }

  async loadGeometry() {
    const meshSource = this.getAttribute("mesh-src") || "data/brain_mesh.json";
    const tractSource = this.getAttribute("tracts-src") || "data/tracts_preview.json";
    const headSource = this.getAttribute("head-src");
    const [mesh, tracts, head] = await Promise.all([
      fetch(new URL(meshSource, document.baseURI)).then((response) => response.json()),
      fetch(new URL(tractSource, document.baseURI)).then((response) => response.json()),
      headSource ? fetch(new URL(headSource, document.baseURI)).then((response) => response.json()) : Promise.resolve(null),
    ]);
    this.geometry = this.prepareGeometry(mesh, tracts, head);
  }

  readSettings() {
    this.alpha = this.readAlpha("alpha", 1);
    const meshA = Number.parseFloat(this.getAttribute("mesh-alpha"));
    this.meshAlpha = Number.isFinite(meshA) ? Math.max(0, meshA) : 1;   // allow >1 to brighten the mesh (title)
    this.keepFinalMesh = this.hasAttribute("keep-mesh");   // recap keeps the surface at the final stage; build strips it
    this.tractAlpha = this.readAlpha("tract-alpha", 1);
    this.nodeAlpha = this.readAlpha("node-alpha", 1);
    this.pulseAlpha = this.readAlpha("pulse-alpha", 1);
    this.fieldAlpha = this.readAlpha("field-alpha", 1);
    const cx = Number.parseFloat(this.getAttribute("center-x"));
    const cy = Number.parseFloat(this.getAttribute("center-y"));
    this.centerXRatio = Number.isFinite(cx) ? cx : null;   // null -> default (right-shifted) layout
    this.centerYRatio = Number.isFinite(cy) ? cy : null;
    // staged reveal for the chapter-1 build-up; default = everything visible (title background)
    this.staged = this.hasAttribute("stage");   // build slide spins; title keeps original motion
    this.reveal = { head: 1, mesh: 1, tract: 1, field: 1, pulse: 1, node: 1, graphEdge: 1, graphPulse: 1 };
    this.revealTarget = { ...this.reveal };
    if (this.hasAttribute("stage")) {
      this.applyStage(Number.parseInt(this.getAttribute("stage"), 10) || 0);
      this.reveal = { ...this.revealTarget };   // start exactly at the initial stage
    }
  }

  // stage 1 head · 2 brain inside · 3 +tracts · 4 +dynamics · 5 +nodes · 6 strip mesh+tracts → network only
  applyStage(n) {
    const s = Math.max(0, Math.min(6, Number.isFinite(n) ? n : 0));
    this.revealTarget = {
      head: s === 1 ? 1 : 0,            // head shown first, then peeled away to reveal the brain inside
      mesh: s >= 2 && (s < 6 || this.keepFinalMesh) ? 1 : 0,   // final stage strips the cortical mesh (unless keep-mesh) → network only
      tract: s >= 3 && s < 6 ? 1 : 0,
      field: s >= 4 ? 1 : 0,
      pulse: s >= 4 && s < 6 ? 1 : 0,
      node: s >= 5 ? 1 : 0,
      graphEdge: s >= 6 ? 1 : 0,
      graphPulse: 0,   // final graph: static edges + pulsating nodes, no travelling particles
    };
  }

  setStage(n) {
    this.applyStage(n);
  }

  readAlpha(attributeName, fallback) {
    const value = Number.parseFloat(this.getAttribute(attributeName));
    if (!Number.isFinite(value)) {
      return fallback;
    }
    return Math.max(0, Math.min(1, value));
  }

  layerAlpha(value, layerAlpha = 1) {
    return Math.max(0, Math.min(1, value * this.alpha * layerAlpha));
  }

  resize() {
    const rect = this.getBoundingClientRect();
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    this.width = Math.max(1, Math.floor(rect.width));
    this.height = Math.max(1, Math.floor(rect.height));
    this.canvas.width = Math.floor(this.width * this.pixelRatio);
    this.canvas.height = Math.floor(this.height * this.pixelRatio);
    this.context.setTransform(this.pixelRatio, 0, 0, this.pixelRatio, 0, 0);
  }

  updatePointer(event) {
    this.pointer.x = (event.clientX / window.innerWidth - 0.5) * 2;
    this.pointer.y = (event.clientY / window.innerHeight - 0.5) * 2;
  }

  prepareGeometry(mesh, tracts, head) {
    const bounds = this.combinedBounds(mesh.vertices, tracts.bounds);
    const center = bounds.min.map((value, index) => (value + bounds.max[index]) / 2);
    const scale = Math.max(...bounds.max.map((value, index) => value - bounds.min[index]));
    const normalize = ([x, y, z]) => [
      (x - center[0]) / scale,
      (z - center[2]) / scale,
      (y - center[1]) / scale,
    ];

    const nodes = (mesh.nodes && mesh.nodes.length ? mesh.nodes : mesh.vertices.filter((_, index) => index % 9 === 0)).map(normalize);

    let headVertices = null;
    let headFaces = null;
    if (head) {
      // head shares the brain's CENTRE (so it stays aligned) but uses its own scale: the full head fits
      // at stage 1, then the brain zooms up to fill the frame on later stages (head → brain = zoom-in)
      let hr = 1e-6;
      for (const v of head.vertices) {
        hr = Math.max(hr, Math.abs(v[0] - center[0]), Math.abs(v[1] - center[1]), Math.abs(v[2] - center[2]));
      }
      const hScale = 2 * hr;
      headVertices = head.vertices.map(([x, y, z]) => [(x - center[0]) / hScale, (z - center[2]) / hScale, (y - center[1]) / hScale]);
      headFaces = head.faces.filter((_, index) => index % 2 === 0);
    }
    return {
      vertices: mesh.vertices.map(normalize),
      faces: mesh.faces.filter((_, index) => index % 2 === 0),
      nodes,
      streamlines: tracts.streamlines.map((streamline) => streamline.map(normalize)),
      graphEdges: (mesh.edges || []).map(([a, b]) => [nodes[a], nodes[b]]).filter((edge) => edge[0] && edge[1]),
      headVertices,
      headFaces,
    };
  }

  boundsOf(vertices) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const vertex of vertices) {
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], vertex[axis]);
        max[axis] = Math.max(max[axis], vertex[axis]);
      }
    }
    return { min, max };
  }

  combinedBounds(vertices, tractBounds) {
    const min = [...tractBounds.min];
    const max = [...tractBounds.max];
    for (const vertex of vertices) {
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], vertex[axis]);
        max[axis] = Math.max(max[axis], vertex[axis]);
      }
    }
    return { min, max };
  }

  render(time) {
    if (!this.geometry || document.visibilityState === "hidden") {
      this.frameRequest = requestAnimationFrame(this.render);
      return;
    }

    const ease = 0.05;   // smooth fade between stages
    for (const key in this.reveal) {
      this.reveal[key] += (this.revealTarget[key] - this.reveal[key]) * ease;
    }

    this.context.clearRect(0, 0, this.width, this.height);
    this.drawFieldLines(time);
    this.drawHead(time);
    this.drawMesh(time);
    this.drawTracts(time);
    this.drawGraphEdges(time);
    this.drawNodes(time);

    if (!this.prefersReducedMotion) {
      this.frameRequest = requestAnimationFrame(this.render);
    }
  }

  drawFieldLines(time) {
    if (this.staged) return;   // build slide shows firing via pulses, not the horizontal field grid
    const lineCount = Math.max(7, Math.floor(this.height / 82));
    this.context.save();
    this.context.lineWidth = 1;
    for (let index = 0; index < lineCount; index += 1) {
      const y = (index + 0.5) * (this.height / lineCount);
      const drift = Math.sin(time * 0.00022 + index) * 18;
      const alpha = this.layerAlpha(0.018 + index * 0.002, this.fieldAlpha) * this.reveal.field;
      this.context.strokeStyle = `rgba(255,255,255,${alpha})`;
      this.context.beginPath();
      this.context.moveTo(this.width * 0.04, y + drift);
      this.context.bezierCurveTo(this.width * 0.32, y - 32, this.width * 0.68, y + 34, this.width * 0.98, y - drift);
      this.context.stroke();
    }
    this.context.restore();
  }

  drawHead(time) {
    if (!this.geometry.headFaces || this.reveal.head < 0.01) {
      return;
    }
    this.context.save();
    this.context.lineWidth = this.width < 720 ? 0.4 : 0.55;
    for (const face of this.geometry.headFaces) {
      const points = face.map((vertexIndex) => this.project(this.geometry.headVertices[vertexIndex], time));
      const alpha = this.layerAlpha(0.05 + Math.max(0, points[0].depth + points[1].depth + points[2].depth) * 0.006, 1) * this.reveal.head;
      this.context.strokeStyle = `rgba(208,202,192,${alpha})`;   // warm scalp grey
      this.context.beginPath();
      this.context.moveTo(points[0].x, points[0].y);
      this.context.lineTo(points[1].x, points[1].y);
      this.context.lineTo(points[2].x, points[2].y);
      this.context.closePath();
      this.context.stroke();
    }
    this.context.restore();
  }

  drawMesh(time) {
    this.context.save();
    this.context.lineWidth = this.width < 720 ? 0.45 : 0.65;
    for (const face of this.geometry.faces) {
      const points = face.map((vertexIndex) => this.project(this.geometry.vertices[vertexIndex], time));
      const alpha = this.layerAlpha(0.032 + Math.max(0, points[0].depth + points[1].depth + points[2].depth) * 0.006, this.meshAlpha) * this.reveal.mesh;
      this.context.strokeStyle = `rgba(108,230,212,${alpha})`;
      this.context.beginPath();
      this.context.moveTo(points[0].x, points[0].y);
      this.context.lineTo(points[1].x, points[1].y);
      this.context.lineTo(points[2].x, points[2].y);
      this.context.closePath();
      this.context.stroke();
    }
    this.context.restore();
  }

  drawTracts(time) {
    this.context.save();
    this.context.lineCap = "round";
    this.context.lineJoin = "round";

    this.geometry.streamlines.forEach((streamline, index) => {
      const color = this.palette[index % this.palette.length];
      const projected = streamline.map((point) => this.project(point, time));
      const alpha = this.layerAlpha(0.11 + (index % 7) * 0.008, this.tractAlpha) * this.reveal.tract;
      this.context.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
      this.context.lineWidth = index % 11 === 0 ? 1.35 : 0.74;
      this.context.beginPath();
      projected.forEach((point, pointIndex) => {
        if (pointIndex === 0) {
          this.context.moveTo(point.x, point.y);
        } else {
          this.context.lineTo(point.x, point.y);
        }
      });
      this.context.stroke();

      if (index % 5 === 0) {
        this.drawPulse(projected, color, (time * 0.00028 + index * 0.071) % 1);
      }
    });

    this.context.restore();
  }

  drawPulse(points, color, phase) {
    const scaled = phase * (points.length - 1);
    const index = Math.min(points.length - 2, Math.floor(scaled));
    const local = scaled - index;
    const start = points[index];
    const end = points[index + 1];
    const x = start.x + (end.x - start.x) * local;
    const y = start.y + (end.y - start.y) * local;
    const alpha = this.layerAlpha(0.9, this.pulseAlpha) * this.reveal.pulse;
    if (alpha < 0.01) return;
    // a small oval (comet) gently elongated along the fibre direction
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    const rx = this.width < 720 ? 1.6 : 2.2;   // long axis (along the fibre)
    const ry = this.width < 720 ? 0.9 : 1.2;   // short axis
    this.context.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
    this.context.beginPath();
    this.context.ellipse(x, y, rx, ry, angle, 0, Math.PI * 2);
    this.context.fill();
  }

  drawNodes(time) {
    this.context.save();
    this.geometry.nodes.forEach((node, index) => {
      const point = this.project(node, time);
      const activity = this.layerAlpha(0.45 + Math.sin(time * 0.0022 + index * 0.7) * 0.28, this.nodeAlpha) * this.reveal.node;
      const color = this.palette[index % this.palette.length];
      this.context.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${activity})`;
      this.context.beginPath();
      this.context.arc(point.x, point.y, this.width < 720 ? 3.4 : 4.8, 0, Math.PI * 2);
      this.context.fill();
    });
    this.context.restore();
  }

  drawGraphEdges(time) {
    if (!this.geometry.graphEdges || !this.geometry.graphEdges.length || this.reveal.graphEdge < 0.01) {
      return;
    }
    this.context.save();
    this.context.lineWidth = this.width < 720 ? 0.5 : 0.7;
    this.geometry.graphEdges.forEach((edge, index) => {
      const a = this.project(edge[0], time);
      const b = this.project(edge[1], time);
      const color = this.palette[index % this.palette.length];
      const alpha = this.layerAlpha(0.14 + Math.max(0, a.depth + b.depth) * 0.05, 1) * this.reveal.graphEdge;
      this.context.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
      this.context.beginPath();
      this.context.moveTo(a.x, a.y);
      this.context.lineTo(b.x, b.y);
      this.context.stroke();
      if (index % 3 === 0) {
        this.drawGraphPulse(a, b, color, (time * 0.0004 + index * 0.137) % 1);
      }
    });
    this.context.restore();
  }

  drawGraphPulse(a, b, color, phase) {
    const alpha = this.layerAlpha(0.85, 1) * this.reveal.graphPulse;
    if (alpha < 0.01) return;
    const x = a.x + (b.x - a.x) * phase;
    const y = a.y + (b.y - a.y) * phase;
    this.context.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
    this.context.beginPath();
    this.context.arc(x, y, this.width < 720 ? 1.5 : 2.1, 0, Math.PI * 2);
    this.context.fill();
  }

  project(point, time) {
    const slowTime = this.prefersReducedMotion ? 1200 : time;
    const spin = this.staged ? slowTime * 0.00014 : 0;   // continuous turntable on the build slide only
    const angleY = -0.42 + spin + Math.sin(slowTime * 0.00013) * (this.staged ? 0.06 : 0.12) + this.pointer.x * 0.08;
    const angleX = 0.28 + Math.sin(slowTime * 0.00009) * 0.06 - this.pointer.y * 0.04;
    const angleZ = -0.08 + Math.sin(slowTime * 0.00011) * 0.04;
    let [x, y, z] = point;

    [x, z] = this.rotatePair(x, z, angleY);
    [y, z] = this.rotatePair(y, z, angleX);
    [x, y] = this.rotatePair(x, y, angleZ);

    const perspective = 1 / (2.45 - z * 0.72);
    const size = Math.min(this.width, this.height) * (this.width < 720 ? 1.34 : 1.72);
    const centerX = (this.centerXRatio != null ? this.centerXRatio : (this.width < 860 ? 0.5 : 0.66)) * this.width;
    const centerY = (this.centerYRatio != null ? this.centerYRatio : (this.width < 860 ? 0.48 : 0.52)) * this.height;

    return {
      x: centerX + x * perspective * size,
      y: centerY - y * perspective * size,
      depth: z,
    };
  }

  rotatePair(first, second, angle) {
    const sin = Math.sin(angle);
    const cos = Math.cos(angle);
    return [first * cos - second * sin, first * sin + second * cos];
  }
}

customElements.define("brain-network-visual", BrainNetworkVisual);