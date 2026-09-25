# labs/ — interactive visualizations (version 1)

Browser pages made while exploring the idea. They run locally with no build
step: open `index.html`. The text is mostly in Russian.

| Page | What it shows |
|---|---|
| `index.html` | Landing page that links the others |
| `quantsim3d.html` | 3D view of the rotation and quantization pipeline, golden angle vs random rotation |
| `formula_lab.html` | Formula editor with a live preview: TurboQuant vs NautilusQuant |
| `charts.html` | Research charts of the early experiments |
| `mirofish_lab.html` | Chat front end for discussing quantizer settings with LLM agents; needs a separate local backend (`localhost:5001`) that is not part of this repository |
| `compute_worker.js` | Web Worker with the rotation and quantizer used by the pages |

These pages illustrate the version 1 algorithm. The findings that followed
(golden angle vs random rotation, the chip, the NQX-RN codec) are in the
[root README](../README.md).
