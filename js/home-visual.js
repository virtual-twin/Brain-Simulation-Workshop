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
    const [mesh, tracts] = await Promise.all([
      fetch(new URL(meshSource, document.baseURI)).then((response) => response.json()),
      fetch(new URL(tractSource, document.baseURI)).then((response) => response.json()),
    ]);
    this.geometry = this.prepareGeometry(mesh, tracts);
  }

  readSettings() {
    this.alpha = this.readAlpha("alpha", 1);
    this.meshAlpha = this.readAlpha("mesh-alpha", 1);
    this.tractAlpha = this.readAlpha("tract-alpha", 1);
    this.nodeAlpha = this.readAlpha("node-alpha", 1);
    this.pulseAlpha = this.readAlpha("pulse-alpha", 1);
    this.fieldAlpha = this.readAlpha("field-alpha", 1);
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

  prepareGeometry(mesh, tracts) {
    const bounds = this.combinedBounds(mesh.vertices, tracts.bounds);
    const center = bounds.min.map((value, index) => (value + bounds.max[index]) / 2);
    const scale = Math.max(...bounds.max.map((value, index) => value - bounds.min[index]));
    const normalize = ([x, y, z]) => [
      (x - center[0]) / scale,
      (z - center[2]) / scale,
      (y - center[1]) / scale,
    ];

    return {
      vertices: mesh.vertices.map(normalize),
      faces: mesh.faces.filter((_, index) => index % 2 === 0),
      nodes: mesh.vertices.filter((_, index) => index % 9 === 0).map(normalize),
      streamlines: tracts.streamlines.map((streamline) => streamline.map(normalize)),
    };
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

    this.context.clearRect(0, 0, this.width, this.height);
    this.drawFieldLines(time);
    this.drawMesh(time);
    this.drawTracts(time);
    this.drawNodes(time);

    if (!this.prefersReducedMotion) {
      this.frameRequest = requestAnimationFrame(this.render);
    }
  }

  drawFieldLines(time) {
    const lineCount = Math.max(7, Math.floor(this.height / 82));
    this.context.save();
    this.context.lineWidth = 1;
    for (let index = 0; index < lineCount; index += 1) {
      const y = (index + 0.5) * (this.height / lineCount);
      const drift = Math.sin(time * 0.00022 + index) * 18;
      const alpha = this.layerAlpha(0.018 + index * 0.002, this.fieldAlpha);
      this.context.strokeStyle = `rgba(255,255,255,${alpha})`;
      this.context.beginPath();
      this.context.moveTo(this.width * 0.04, y + drift);
      this.context.bezierCurveTo(this.width * 0.32, y - 32, this.width * 0.68, y + 34, this.width * 0.98, y - drift);
      this.context.stroke();
    }
    this.context.restore();
  }

  drawMesh(time) {
    this.context.save();
    this.context.lineWidth = this.width < 720 ? 0.45 : 0.65;
    for (const face of this.geometry.faces) {
      const points = face.map((vertexIndex) => this.project(this.geometry.vertices[vertexIndex], time));
      const alpha = this.layerAlpha(0.032 + Math.max(0, points[0].depth + points[1].depth + points[2].depth) * 0.006, this.meshAlpha);
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
      const alpha = this.layerAlpha(0.11 + (index % 7) * 0.008, this.tractAlpha);
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
    const radius = this.width < 720 ? 1.7 : 2.4;
    const alpha = this.layerAlpha(0.86, this.pulseAlpha);
    this.context.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
    this.context.beginPath();
    this.context.arc(x, y, radius, 0, Math.PI * 2);
    this.context.fill();
  }

  drawNodes(time) {
    this.context.save();
    this.geometry.nodes.forEach((node, index) => {
      const point = this.project(node, time);
      const activity = this.layerAlpha(0.45 + Math.sin(time * 0.0022 + index * 0.7) * 0.28, this.nodeAlpha);
      const color = this.palette[index % this.palette.length];
      this.context.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${activity})`;
      this.context.beginPath();
      this.context.arc(point.x, point.y, this.width < 720 ? 0.9 : 1.25, 0, Math.PI * 2);
      this.context.fill();
    });
    this.context.restore();
  }

  project(point, time) {
    const slowTime = this.prefersReducedMotion ? 1200 : time;
    const angleY = -0.42 + Math.sin(slowTime * 0.00013) * 0.12 + this.pointer.x * 0.08;
    const angleX = 0.28 + Math.sin(slowTime * 0.00009) * 0.06 - this.pointer.y * 0.04;
    const angleZ = -0.08 + Math.sin(slowTime * 0.00011) * 0.04;
    let [x, y, z] = point;

    [x, z] = this.rotatePair(x, z, angleY);
    [y, z] = this.rotatePair(y, z, angleX);
    [x, y] = this.rotatePair(x, y, angleZ);

    const perspective = 1 / (2.45 - z * 0.72);
    const size = Math.min(this.width, this.height) * (this.width < 720 ? 1.34 : 1.72);
    const centerX = this.width < 860 ? this.width * 0.5 : this.width * 0.66;
    const centerY = this.height * (this.width < 860 ? 0.48 : 0.52);

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