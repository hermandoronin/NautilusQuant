// ============================================================
// Compute Worker v2 — все 7 багов исправлены
// Fix 4: Butterfly с непересекающимися парами (odd-even)
// Fix 5: Обратное преобразование T^(-1) для dequant
// Fix 7: Квантование идентично TurboQuant (нет predicted_scale)
// ============================================================

const PHI = (1 + Math.sqrt(5)) / 2;
const PI = Math.PI;
const TAU = 2 * PI;

function rng(s) {
  return () => {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ============ GIVENS ROTATION (атомарная операция) ============
// Вращает ТОЛЬКО пару (i,j) на угол theta. Ортогонально по определению.
function givensRotate(v, i, j, theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  const a = v[i], b = v[j];
  v[i] = a * c - b * s;
  v[j] = a * s + b * c;
}

// Обратное вращение: тот же Givens, но с -theta
function givensRotateInverse(v, i, j, theta) {
  givensRotate(v, i, j, -theta);
}

// ============ ГЕНЕРАЦИЯ СЛОЁВ (Fix 4: непересекающиеся пары) ============

// Генерирует массив слоёв. Каждый слой = [{i, j, theta}, ...]
// Гарантия: в одном слое каждый индекс участвует РОВНО один раз
function buildNautilusLayers(dim, phi) {
  phi = phi || PHI;
  const ga = TAU / (phi * phi); // golden angle
  const layers = [];

  // --- Слой 1: соседние пары (0,1), (2,3), (4,5)... ---
  // Все пары непересекающиеся ✓
  const layer1 = [];
  for (let k = 0; k < Math.floor(dim / 2); k++) {
    const i = 2 * k, j = 2 * k + 1;
    const theta = ga * (k + 1);
    layer1.push({ i, j, theta });
  }
  layers.push(layer1);

  // --- Слой 2: сдвинутые пары (1,2), (3,4), (5,6)... ---
  // Все пары непересекающиеся ✓
  const layer2 = [];
  for (let k = 0; k < Math.floor((dim - 1) / 2); k++) {
    const i = 2 * k + 1, j = 2 * k + 2;
    const theta = ga * (k + 1) * phi;
    layer2.push({ i, j, theta });
  }
  layers.push(layer2);

  // --- Слой 3: butterfly (Fix 4!) ---
  // Odd-even pattern: строго непересекающиеся дальние пары
  // Берём stride = dim/4, строим пары так, чтобы каждый индекс использовался ОДИН раз
  const layer3 = [];
  const stride = Math.max(2, Math.floor(dim / 4));
  const used = new Set();
  for (let k = 0; k < dim; k++) {
    const i = k;
    const j = (k + stride) % dim;
    if (i === j) continue;
    if (used.has(i) || used.has(j)) continue; // ← FIX 4: пропускаем пересечения!
    used.add(i);
    used.add(j);
    const theta = ga * (k + 1) * phi * phi;
    layer3.push({ i, j, theta });
  }
  layers.push(layer3);

  return layers;
}

// ============ ПРЯМОЕ ПРЕОБРАЗОВАНИЕ (Encode) ============
function nautilusForward(v, layers) {
  const o = [...v];
  // Применяем слои по порядку: 1 → 2 → 3
  for (const layer of layers) {
    for (const { i, j, theta } of layer) {
      givensRotate(o, i, j, theta);
    }
  }
  return o;
}

// ============ ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ (Fix 5: Decode) ============
// T^(-1) = Layer1^T · Layer2^T · Layer3^T
// Каждый Layer^T = те же вращения с -theta, в обратном порядке внутри слоя
function nautilusInverse(v, layers) {
  const o = [...v];
  // Слои в обратном порядке: 3 → 2 → 1
  for (let l = layers.length - 1; l >= 0; l--) {
    const layer = layers[l];
    // Вращения внутри слоя в обратном порядке с -theta
    for (let r = layer.length - 1; r >= 0; r--) {
      givensRotateInverse(o, layer[r].i, layer[r].j, layer[r].theta);
    }
  }
  return o;
}

// ============ ПРОВЕРКА ОРТОГОНАЛЬНОСТИ ============
function verifyOrthogonality(dim, layers, sampleVec) {
  const forward = nautilusForward(sampleVec, layers);
  const roundtrip = nautilusInverse(forward, layers);

  // Проверка 1: |T·v| == |v| (сохранение нормы)
  const origNorm = Math.sqrt(sampleVec.reduce((s, x) => s + x * x, 0));
  const fwdNorm = Math.sqrt(forward.reduce((s, x) => s + x * x, 0));
  const normError = Math.abs(origNorm - fwdNorm);

  // Проверка 2: T^(-1)·T·v == v (roundtrip)
  let roundtripError = 0;
  for (let i = 0; i < dim; i++) {
    roundtripError += (sampleVec[i] - roundtrip[i]) ** 2;
  }
  roundtripError = Math.sqrt(roundtripError);

  // Проверка 3: dot(T·a, T·b) == dot(a, b) (сохранение скалярного произведения)
  const r = rng(99);
  const vecB = Array.from({ length: dim }, () => r() * 2 - 1);
  const fwdB = nautilusForward(vecB, layers);
  const origDot = sampleVec.reduce((s, x, i) => s + x * vecB[i], 0);
  const fwdDot = forward.reduce((s, x, i) => s + x * fwdB[i], 0);
  const dotError = Math.abs(origDot - fwdDot);

  return {
    normError,        // должно быть ~0 (< 1e-10)
    roundtripError,   // должно быть ~0 (< 1e-10)
    dotProductError: dotError, // должно быть ~0 (< 1e-10)
    isOrthogonal: normError < 1e-8 && roundtripError < 1e-8 && dotError < 1e-8
  };
}

// ============ PIPELINE FUNCTIONS ============

function generateVectors(count, dim, seed) {
  const r = rng(seed || 42);
  const vecs = [];
  for (let i = 0; i < count; i++) {
    const v = [];
    for (let d = 0; d < dim; d++) {
      let val = r() * 2 - 1;
      if (r() < 0.06) val *= 7; // outliers
      v.push(val);
    }
    vecs.push(v);
  }
  return vecs;
}

// TurboQuant: случайные углы (baseline)
function turboRotate(vecs, seed) {
  const dim = vecs[0].length;
  return vecs.map(v => {
    const o = [...v];
    const r = rng(seed || 77);
    for (let i = 0; i < dim - 1; i += 2) {
      givensRotate(o, i, i + 1, r() * TAU);
    }
    return o;
  });
}

// NautilusQuant v2: золотые углы, строго ортогонально
function nautilusRotate(vecs, phi) {
  const dim = vecs[0].length;
  const layers = buildNautilusLayers(dim, phi);
  return vecs.map(v => nautilusForward(v, layers));
}

// Для dequant (Fix 5)
function nautilusUnrotate(vecs, phi) {
  const dim = vecs[0].length;
  const layers = buildNautilusLayers(dim, phi);
  return vecs.map(v => nautilusInverse(v, layers));
}

function toPolar(vecs) {
  return vecs.map(v => {
    const p = [];
    for (let i = 0; i < v.length - 1; i += 2) {
      p.push(Math.sqrt(v[i] * v[i] + v[i + 1] * v[i + 1]));
      p.push(Math.atan2(v[i + 1], v[i]));
    }
    if (v.length % 2) p.push(v[v.length - 1]);
    return p;
  });
}

// Fix 7: квантование ИДЕНТИЧНО для обоих методов. Нет predicted_scale.
// Overhead одинаковый — честное сравнение.
function quantize(vecs, bits) {
  const lv = 1 << bits, d = vecs[0].length;
  const mn = new Float64Array(d).fill(1e9);
  const mx = new Float64Array(d).fill(-1e9);
  for (const v of vecs) for (let i = 0; i < d; i++) {
    mn[i] = Math.min(mn[i], v[i]);
    mx[i] = Math.max(mx[i], v[i]);
  }
  const overheadBitsPerDim = 32; // 2×FP16 на scale+zp (ЧЕСТНО для обоих!)
  return {
    quantized: vecs.map(v => v.map((x, i) => {
      const range = mx[i] - mn[i] || 1;
      const q = Math.round((x - mn[i]) / range * (lv - 1));
      return q / (lv - 1) * range + mn[i];
    })),
    overhead: overheadBitsPerDim,
    mins: mn,
    maxs: mx
  };
}

function qjlCorrect(orig, quant, alpha) {
  alpha = alpha || 0.5;
  return orig.map((v, i) => v.map((x, j) => {
    const err = x - quant[i][j];
    return quant[i][j] + (err >= 0 ? 1 : -1) * Math.abs(err) * alpha;
  }));
}

function computeMSE(a, b) {
  let sum = 0, count = 0;
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a[i].length; j++) {
      sum += (a[i][j] - b[i][j]) ** 2;
      count++;
    }
  return sum / count;
}

// ============ МЕТРИКИ РАСПРЕДЕЛЕНИЯ (для доказательства пункта 1 и 3) ============

function analyzeDistribution(polarVecs) {
  // Анализ распределения углов после полярного преобразования
  // Если дисперсия углов ниже → распределение предсказуемее → квантизатор точнее
  const allAngles = [];
  const allRadii = [];
  for (const v of polarVecs) {
    for (let i = 1; i < v.length; i += 2) allAngles.push(v[i]); // theta
    for (let i = 0; i < v.length; i += 2) allRadii.push(v[i]);   // r
  }

  // Гистограмма углов (32 бина)
  const bins = 32;
  const angleBins = new Float64Array(bins);
  for (const a of allAngles) {
    angleBins[Math.min(bins - 1, Math.floor((a + PI) / TAU * bins))]++;
  }
  const mean = allAngles.length / bins;
  let angleVariance = 0;
  for (const b of angleBins) angleVariance += (b - mean) ** 2;
  angleVariance /= bins;

  // Разброс радиусов
  const rMean = allRadii.reduce((s, x) => s + x, 0) / allRadii.length;
  let rVariance = 0;
  for (const r of allRadii) rVariance += (r - rMean) ** 2;
  rVariance /= allRadii.length;

  // Диапазон (нужен ли overhead?)
  const rMin = Math.min(...allRadii), rMax = Math.max(...allRadii);
  const aMin = Math.min(...allAngles), aMax = Math.max(...allAngles);

  return {
    angleVariance,     // ниже = лучше (предсказуемее)
    radiusVariance: rVariance,
    radiusMean: rMean,
    radiusRange: [rMin, rMax],
    angleRange: [aMin, aMax],
    angleBins: Array.from(angleBins),
    totalAngles: allAngles.length,
    totalRadii: allRadii.length
  };
}

// ============ FULL PIPELINE ============

function runFullPipeline(params) {
  const { count, dim, bits, phi, seed } = params;
  const vecs = generateVectors(count, dim, seed || 42);
  const layers = buildNautilusLayers(dim, phi || PHI);

  // --- Проверка ортогональности (Fix 4 validation) ---
  const orthCheck = verifyOrthogonality(dim, layers, vecs[0]);

  // --- TurboQuant pipeline ---
  const tRot = turboRotate(vecs, 77);
  const tPol = toPolar(tRot);
  const tQ = quantize(tPol, bits);
  const tQjl = qjlCorrect(tPol, tQ.quantized);
  const tMSE = computeMSE(tPol, tQjl);
  const tDist = analyzeDistribution(tPol);

  // --- NautilusQuant v2 pipeline ---
  const nRot = nautilusRotate(vecs, phi || PHI);
  const nPol = toPolar(nRot);
  const nQ = quantize(nPol, bits);
  const nQjl = qjlCorrect(nPol, nQ.quantized);
  const nMSE = computeMSE(nPol, nQjl);
  const nDist = analyzeDistribution(nPol);

  // --- Fix 5: roundtrip test ---
  const nReconstructed = nautilusUnrotate(nRot, phi || PHI);
  let roundtripMSE = 0;
  for (let i = 0; i < Math.min(10, vecs.length); i++) {
    for (let j = 0; j < dim; j++) {
      roundtripMSE += (vecs[i][j] - nReconstructed[i][j]) ** 2;
    }
  }
  roundtripMSE /= Math.min(10, vecs.length) * dim;

  // --- Fix 7: overhead ЧЕСТНЫЙ для обоих ---
  const overheadPerGroup = tQ.overhead; // одинаковый!

  return {
    turbo: {
      mse: tMSE,
      overhead: overheadPerGroup,
      distribution: tDist,
      sampleRotated: tRot.slice(0, 20),
      sampleQuant: tQjl.slice(0, 20)
    },
    nautilus: {
      mse: nMSE,
      overhead: overheadPerGroup, // Fix 7: честно — такой же!
      distribution: nDist,
      sampleRotated: nRot.slice(0, 20),
      sampleQuant: nQjl.slice(0, 20)
    },
    orthogonality: orthCheck,
    roundtripMSE,
    compression: 16 / bits,
    inputSample: vecs.slice(0, 20),
    // Главный вопрос: если angleVariance у Nautilus ниже → золотые углы ЛУЧШЕ
    nautilusAdvantage: {
      angleVarianceDiff: tDist.angleVariance - nDist.angleVariance,
      mseDiff: tMSE - nMSE,
      nautilusBetter: nMSE < tMSE
    }
  };
}

// Sweep PHI values
function sweepPhi(params) {
  const { count, dim, bits, phiValues } = params;
  const vecs = generateVectors(count, dim, 42);
  const results = [];

  for (const phi of phiValues) {
    const layers = buildNautilusLayers(dim, phi);
    const nRot = vecs.map(v => nautilusForward(v, layers));
    const nPol = toPolar(nRot);
    const nQ = quantize(nPol, bits);
    const nQjl = qjlCorrect(nPol, nQ.quantized);
    const mse = computeMSE(nPol, nQjl);
    const dist = analyzeDistribution(nPol);
    const orth = verifyOrthogonality(dim, layers, vecs[0]);

    results.push({
      phi,
      mse,
      angleVariance: dist.angleVariance,
      isOrthogonal: orth.isOrthogonal,
      normError: orth.normError,
      label: getPhiLabel(phi)
    });
  }

  return results;
}

function getPhiLabel(phi) {
  if (Math.abs(phi - (1 + Math.sqrt(5)) / 2) < 0.001) return 'φ (Golden Ratio)';
  if (Math.abs(phi - Math.PI) < 0.001) return 'π';
  if (Math.abs(phi - Math.E) < 0.001) return 'e (Euler)';
  if (Math.abs(phi - Math.sqrt(2)) < 0.001) return '√2';
  if (Math.abs(phi - Math.sqrt(3)) < 0.001) return '√3';
  return phi.toFixed(4);
}

// ============ MESSAGE HANDLER ============
self.onmessage = function (e) {
  const { id, action, params } = e.data;
  let result;
  try {
    switch (action) {
      case 'runPipeline': result = runFullPipeline(params); break;
      case 'sweepPhi': result = sweepPhi(params); break;
      case 'verify': {
        const dim = params.dim || 16;
        const phi = params.phi || PHI;
        const layers = buildNautilusLayers(dim, phi);
        const r2 = rng(42);
        const testVec = Array.from({ length: dim }, () => r2() * 2 - 1);
        result = verifyOrthogonality(dim, layers, testVec);
        break;
      }
      case 'customFormula': {
        const fn = new Function(...params.argNames, params.code);
        result = fn(...params.argValues);
        break;
      }
      default: result = { error: 'Unknown action: ' + action };
    }
  } catch (err) {
    result = { error: err.message };
  }
  self.postMessage({ id, result });
};
