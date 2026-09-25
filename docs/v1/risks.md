# NautilusQuant — Risk Analysis & Failure Modes

> Честный анализ: что может сломаться и что делать когда сломается.

---

## 3 фундаментальных риска

### Risk 1: Структурный резонанс с выбросами

**Проблема:** В трансформерах систематические выбросы (до −60 при фоне ±0.5) сконцентрированы в **6 конкретных измерениях**, активных в 75% позиций. Матрица NautilusQuant фиксирована (детерминирована). Если шаги золотой спирали (137.5° × k) неудачно совпадут с этими 6 измерениями, спираль **не размажет, а сконцентрирует** выбросы. MSE взлетит, модель начнёт галлюцинировать.

**TurboQuant защищён:** Случайное вращение работает как миксер — гарантированно размазывает любые паттерны.

**Как проверить:**
```bash
python validate_real_kv.py --model google/gemma-3-4b-it --sweep
# Смотреть: MSE(nautilus) vs MSE(turbo) на РЕАЛЬНЫХ тензорах
# Если MSE(nautilus) >> MSE(turbo) → резонанс подтверждён
```

**Mitigation:** Если резонанс обнаружен — добавить один слой случайной перестановки (permutation) координат ПЕРЕД золотыми вращениями. Перестановка ортогональна и детерминирована (фиксированная таблица), но разрывает структурное совпадение.

**Файл:** `validate_real_kv.py` → `run_pipeline()` покажет MSE и angle variance.

---

### Risk 2: Провал гипотезы «0 overhead»

**Проблема:** Google добивается 0 overhead потому что случайность приводит координаты к предсказуемому Beta-распределению. Мы предполагаем, что золотой угол даёт ещё более предсказуемое распределение (O(1/N) vs O(1/√N)). Если эмпирика покажет что дисперсия углов NautilusQuant **шире** чем у TurboQuant — придётся хранить FP32 scale/zero-point. Overhead ≠ 0. Магия исчезает.

**Как проверить:**
```bash
python validate_real_kv.py --model google/gemma-3-4b-it
# Ключевая метрика: Angle variance (Nautilus) vs Angle variance (Turbo)
# Если Nautilus variance > Turbo variance → гипотеза опровергнута
```

**Mitigation (Plan B):** MX-Format Fallback.
- `nautilus_hardware.py` → `NautilusWithMX` автоматически переключается на MXFP4
- Overhead: 0.25 bit/value (вместо 0) — всё равно лучше чем FP32 scale/zp
- Аппаратный выигрыш (LUT + determinism) сохраняется даже при overhead ≠ 0

**Файл:** `nautilus_hardware.py` → `NautilusWithMX.encode()`

---

### Risk 3: Накопление FP16 ошибок округления

**Проблема:** Обратное преобразование T⁻¹ = L₁ᵀ · L₂ᵀ · L₃ᵀ теоретически точно (T⁻¹·T·v = v). Но GPU считают в FP16/BF16. Три слоя вращений туда + три обратно = много операций multiply-accumulate. Ошибки округления накапливаются. На длинных контекстах (104K токенов) dot products начинают "плыть".

**Как проверить:**
```bash
python validate_real_kv.py --dim 128 --count 1000
# Смотреть: Roundtrip error (должно быть < 1e-6 для FP16)
# Если > 1e-4 → проблема для длинных контекстов
```

**Mitigation:**
1. Kahan summation в критических путях (6 extra FLOPs per rotation)
2. Периодическая ренормализация: каждые 1000 токенов пересчитывать нормы
3. Mixed precision: вращения в FP32, хранение в FP16

**Файл:** `validate_real_kv.py` → Orthogonality Check (norm error, roundtrip error, dot error)

---

## Worst Case: Всё сломалось. Что остаётся?

Даже если ВСЕ 3 риска реализуются одновременно:

| Аспект | Статус |
|---|---|
| 0 overhead | ❌ Нет (используем MX fallback → 0.25 bit/val) |
| MSE лучше TurboQuant | ❌ Нет (золотые углы не лучше случайных) |
| FP16 roundtrip | ❌ Дрейфует (нужен FP32 или Kahan) |

**NautilusQuant ВСЁ РАВНО выигрывает у TurboQuant по:**

1. **Детерминированность** → 100% воспроизводимость, нет зависимости от PRNG seed
2. **Dataflow-совместимость** → идеален для Groq LPU, Cerebras WSE-3 (нет ветвлений)
3. **LUT в 512 байт** → предвычисленные cos/sin в регистрах, быстрее PRNG
4. **Аудируемость** → можно математически доказать свойства матрицы (не "наверное случайность сработает")

---

## Decision Matrix

| Результат эксперимента | Действие | Публикация |
|---|---|---|
| φ лучше random по MSE И 0 overhead | 🏆 Прорыв. Публиковать как new SOTA. | ICLR 2027 / NeurIPS |
| φ ≈ random по MSE, 0 overhead | 📊 Сильный результат: детерминизм + 0 overhead | ML workshop |
| φ ≈ random, overhead ≠ 0 | 🔧 MX fallback + dataflow advantage | Systems paper (MLSys) |
| φ хуже random (резонанс) | 🛠 Добавить permutation layer, пере-тестировать | Tech report → iterate |

---

## Timeline

```
Сейчас:     Код готов, все файлы на GitHub
Следующий:  pip install torch transformers → python validate_real_kv.py --model google/gemma-3-4b-it --sweep
Результат:  Через 30 минут будет ответ на главный вопрос
```

*Дата: 2026-03-27*
