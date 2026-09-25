# Von der Hypothese des goldenen Winkels zum Silizium: deterministische KV-Cache-Kompression in zwei Versionen

**Technischer Preprint, Revision 2 · 24. September 2026 · Herman Doronin**
Repository: <https://github.com/hermandoronin/NautilusQuant>

*Übersetzt aus dem russischen Original ([nautilusquant_v1_v2.ru.md](nautilusquant_v1_v2.ru.md)).*

> **Status.** Logik und Verifikation des Chips sind abgeschlossen (Modell, RTL,
> Simulation auf Pin-Ebene, formale Beweise, 91 Referenztransaktionen).
> Der finale IHP-Sign-off-Lauf ist sauber: DRC mit dem vollständigen
> IHP-Regelsatz — 0 Verletzungen, Metalldichte und Antennen — 0, LVS — die
> Schaltungen stimmen überein, das Timing schließt in allen drei Corners
> (Slack +1,08 ns im Slow Corner). Paket für die Foundry:
> `nqx-silicon/tapeout/ihp-sg13g2/`. Die finale Netlist nach dem Routing
> besteht mit den IHP-Zellmodellen alle 91 Referenztransaktionen Bit für
> Bit.

## Zusammenfassung

Der Cache für Keys und Values (Schlüssel und Werte), kurz KV-Cache, bestimmt
den Speicherbedarf bei der Inferenz großer Sprachmodelle. Die besten
Verfahren zu seiner Kompression drehen die Vektoren mit einer zufälligen
orthogonalen Matrix und quantisieren sie danach. NautilusQuant ersetzt die
zufällige Matrix durch Schichten von Givens-Rotationen mit Winkeln, die
Vielfache des goldenen Winkels 2π/φ² sind. So entsteht eine vollständig
deterministische Transformation, die fast keinen Speicher braucht. Version 1
prüfte diese Idee in Software und im Emulator: Der Rotationszustand schrumpfte
auf eine Tabelle von 1 910 Byte, doch bei der Rekonstruktionsqualität verlor
die Rotation um 7,9 % gegen die zufällige.

Version 2 überträgt den Algorithmus in Silizium. Bei der Umstellung auf
Festkomma zeigte sich: Die dritte Rotationsschicht ist die Identität, und die
Winkel der zweiten Schicht sind die Winkel der ersten mit umgekehrtem
Vorzeichen. Deshalb passt der gesamte Rotationszustand in zwei 32-Bit-Register.
Ein neuer Quantisierer senkt den Rekonstruktionsfehler von 0,316 auf 0,146 bei
denselben 4 Bit pro Wert. Der Testchip NQX-S1 stimmt bitgenau mit dem Modell
überein, sein Schnittstellenprotokoll ist formal bewiesen. Das vollständige
Die mit 2×2 mm ist für den Prozess IHP SG13G2 130 nm entworfen und hat den
Sign-off bestanden: DRC, LVS, Dichte, Antennen und Timing in drei Corners. Eine
Durchmischungsanalyse erklärt den Rückstand des goldenen Winkels: Die Rotation
über benachbarte Paare verteilt jeden Kanal auf nur 4 von 128. Die Arbeit
schließt mit dem Vorschlag einer dritten Version: einem aufgabenspezifischen
Chip-Template, dessen Parameter anhand der Aufgabendaten mit getrennter
Validierung gewählt werden.

Teil II führt eine Rückwärtsanalyse durch: Offene KV-Kompressionsverfahren
(KIVI, KVQuant, QuaRot, TurboQuant, PolarQuant, llama.cpp) und NQX werden auf
einem KV-Cache-Emulator mit RoPE, Kanalausreißern und Attention-Ausgabe
verglichen. Die Butterfly-Rotation mit goldenen Winkeln unterscheidet sich
statistisch nicht von Hadamard-Rotation und zufälliger Rotation, bei null
Multiplizierern und einer Bitgenauigkeit, die eine Float-Rotation nicht hat:
Beim Übergang fp32 → bf16 ändert sich bei 73 % der Vektoren mindestens ein
Code. Ein polarer Code der Keys vor RoPE über RoPE-Paare mit Token-Puffer
erreicht ganz ohne Rotation die Genauigkeit der besten Streaming-Verfahren mit
Rotation.
Die Abstimmung der Winkel auf den Attention-Fehler überträgt sich nicht auf
neue Daten.

Teil III entwickelt den gefundenen Punkt zum Codec NQX-RN weiter. Der Key wird
vor RoPE gespeichert, jedes RoPE-Paar wird als Radius und Winkel abgelegt. Die
Bits werden nach der Query-Energie auf die Paare verteilt, die sich unter RoPE
nicht ändert, und die Position ist ein ganzzahliges 32-Bit-Phasenwort. In der
Emulation ist NQX-RN bei 3,125 Bit pro Wert genauer als Hadamard-Rotation und
zufällige Rotation bei 4,125 Bit, spart also etwa ein Bit pro Wert. Die
Positionen sind bei jeder Kontextlänge exakt. Der Vorteil verschwindet ohne
massive Werte in den Keys und bei einer Verschiebung der Statistik gegenüber
der Kalibrierung um 40 %; beide Bedingungen sind noch an echten Modellen zu
prüfen.

**Schlüsselwörter:** KV-Cache, Quantisierung, Givens-Rotationen, goldener
Winkel, CORDIC, RoPE, polare Quantisierung, anwendungsspezifische ICs, offene
PDKs, IHP SG13G2.

## 1. Einleitung

Bei der Inferenz eines Sprachmodells liest jedes neue Token die Keys und Values
aller vorherigen Token. Für ein Modell mit 7 Mrd. Parametern und einem Kontext
von 128 Tsd. Token belegt der KV-Cache in FP16 etwa 64 GB und wird zum größten
Verbraucher von HBM-Speicher. Seine Kompression erhöht daher direkt den
Durchsatz.

Verfahren wie TurboQuant [1] und QuaRot [2] drehen den Vektor zuerst mit einer
orthogonalen Matrix. Die Rotation erhält Längen und Skalarprodukte, also auch
die Attention-Scores, verschmiert aber Ausreißer über alle Koordinaten. Danach
lässt sich der Vektor gut auf 3–4 Bit quantisieren. Die zufällige Matrix muss
man speichern oder erzeugen: 32 KB in FP16 bei Dimension 128 und 2 MB bei
Dimension 1024. Außerdem passt sie schlecht zu deterministischer Hardware ohne
Zufallszahlengenerator.

Die Hypothese von NautilusQuant: die zufällige Matrix durch ein Produkt von
Givens-Rotationen mit den Winkeln ersetzen

    θ_k = (2π / φ²) · (k + 1) ≈ 137,508° · (k + 1),   φ = (1 + √5) / 2      (1)

Die Folge des goldenen Winkels ist gleichverteilt auf dem Kreis, mit der
kleinstmöglichen Abweichung der Ordnung O(1/N) (Satz von Weyl [6]). Geprüft
wurde, ob diese Gleichmäßigkeit der Winkel zu einem kleineren
Quantisierungsfehler führt.

## 2. Version 1: Software-Prototyp (März – Juli 2026)

### 2.1 Algorithmus

Ein Vektor der Dimension d durchläuft fünf Stufen: drei Schichten von
Givens-Rotationen, Übergang zu Polarkoordinaten über Paare, 3-Bit-Quantisierung,
1-Bit-Restkorrektur nach dem QJL-Schema und Packen auf 4 Bit pro Wert. Die
Schichten wirken auf die Paare L1: (2k, 2k+1), L2: (2k+1, 2k+2), L3: (k, k+d/4);
der Winkel des Paares k in Schicht ℓ ist θ_k·φ^ℓ. Eine frühe Formulierung mit
zentripetaler Skalierung φ^(−i/d) verletzte die Orthogonalität; sie wurde vor
Version 1 entfernt.

### 2.2 Implementierung

- Referenz in PyTorch und Triton sowie Winkeltabelle (ROM mit cos/sin): 1 910 Byte
  bei d = 128, 15 350 Byte bei d = 1024.
- NQX-Core: taktgenauer Emulator in NumPy, Befehlssatz mit 24 Opcodes,
  Assembler, FastAPI-Server, 247 Tests.
- RTL-Gerüst in SystemVerilog mit Platzhaltern statt Arithmetik
  (`polar_unit.sv` berechnete `x ^ y`, wo ein CORDIC hingehört).

### 2.3 Ergebnisse und ihre Revision

Die Orthogonalität bestätigte sich (Fehler TᵀT − I in der Größenordnung
1,6·10⁻⁷), die Ausgabe war Bit für Bit reproduzierbar. Ein direkter Vergleich
mit einer zufälligen orthogonalen Matrix auf synthetischen Daten mit Ausreißern
zeigte, dass der goldene Winkel beim RMSE **um 7,9 % schlechter** ist (0,1429
gegenüber 0,1325). Die Angaben zur Geschwindigkeit (32-mal weniger Takte) und
zur Energie (9,7-mal weniger) stammten aus einem analytischen Modell des
Projekts und waren nicht gemessen. Im Juli 2026 wurden alle veröffentlichten
Zahlen gegen den Code geprüft, gemessene von modellierten getrennt und das
negative Ergebnis an den Anfang der README gestellt.

**Fazit von Version 1:** Vertretbar sind die Kompaktheit des Rotationszustands
und der Determinismus; eine Überlegenheit in der Qualität bestätigte sich
nicht; eine Hardware-Implementierung gibt es nicht.

## 3. Version 2: Testchip NQX-S1 (September 2026)

Ziel ist ein fertigbarer Chip, der die Kompression in Hardware ausführt, Bit
für Bit mit dem Referenzmodell übereinstimmt und ein vollständiges
Dokumentenpaket für die Bestellung bei der Foundry mitbringt. Der Ablauf folgt
industrieller Praxis: bitgenaues Modell → RTL → Simulation und formale
Verifikation → Synthese, Platzierung, Routing → DRC, LVS und Timing-Analyse →
Paket für die Foundry.

### 3.1 Was sich bei der Umstellung auf Festkomma zeigte

    θ_3(k) = (2π/φ²)(k+1)·φ² = 2π(k+1)                                    (2)

| Nr. | Befund | Folge für den Chip |
|---|---|---|
| F1 | Der Winkel der dritten Schicht ist eine ganze Zahl von Umdrehungen (2): L3 ist die Identität und mischt nichts | Die Schicht mit Nullinkrement wird übersprungen; die Kodierung kostet zwei Schichten statt drei, das Ergebnis bleibt gleich |
| F2 | Da 1/φ + 1/φ² = 1 gilt, sind die Winkel von L2 die Winkel von L1 mit umgekehrtem Vorzeichen | Die Phaseninkremente `0x9E3779B9` und `0x61C88647` sind die Konstanten des Fibonacci-Hashings nach Knuth [7] |
| F3 | Die Winkel bilden eine lineare Phasenrampe (k+1)·c mod 2π | Die Tabelle mit 1,9 KB entfällt: ein 32-Bit-Phasenakkumulator pro Schicht, der gesamte Zustand sind 12 Byte Register |
| F4 | Den Fehler bestimmt am stärksten der Quantisierer, nicht die Rotation | Neue Quantisierungsregeln senken den RMSE von 0,316 auf 0,146 bei denselben 4 Bit (§3.3) |

*Tabelle 1.* Details: [`spec/11_algorithm_findings.md`](../spec/11_algorithm_findings.md).

### 3.2 Architektur

Ein Vektor aus 128 int16-Zahlen liegt in einem Registerfile mit 128×24 Bit
(2 Leseports, 2 Schreibports). Die Rotationen führt ein gepipelineter CORDIC [8]
mit Shifts und Additionen aus: 18 Iterationen, Datenpfad 28 Bit, Winkel 20 Bit
pro voller Umdrehung, Latenz 20 Takte, ein Paar pro Takt. Eine iterative
Variante liefert Bit für Bit dasselbe Ergebnis bei 25 % weniger Fläche. Die
Schnittstelle ist ein 8-Bit-Bus mit asynchronem Vier-Phasen-Handshake.

| Mikrooperation | LDV | ROT L1 | ROT L2 | ROT L3 | POLAR | QUANT | **ENC** | **DEC** |
|---|---|---|---|---|---|---|---|---|
| Takte | 258 | 88 | 87 | 3 (übersprungen) | 87 | 200 | **720** | **690** |

*Tabelle 2.* Takte des Befehls ENC (256 Byte → Paket mit 70 Byte, Kompression
3,66×), gemessen in der RTL-Simulation. Bei 50 MHz sind das etwa 69 Tsd.
Vektoren pro Sekunde.

### 3.3 Quantisierer S1

1. **Radiusbereich pro Vektor.** Minimum und Maximum stehen im 6-Byte-Header
   des Pakets; den 3-Bit-Code berechnet ein exakter ganzzahliger Vergleich
   q = #{14d > (2m−1)R}.
2. **Zirkulärer Winkelcode.** 8 Sektoren über den Kreis ohne Sprung bei ±π.
3. **Nutzung des Rests beim Dekodieren.** Das QJL-Bit verschiebt den
   rekonstruierten Wert um ein Viertel der Quantisierungsschrittweite.

### 3.4 Verifikation

| Ebene | Was geprüft wurde | Ergebnis |
|---|---|---|
| Modell | 23 Tests: Konstanten, Paartabellen gegen die Referenz (d von 8 bis 256), CORDIC-Genauigkeit, Rundung | bestanden |
| CORDIC-Block | 4 254 Operationen, Grenz- und Zufallseingaben | Bit für Bit |
| Kern | Zufallsprogramme, unzulässige Opcodes, Busverzögerungen 10–30 % | Bit für Bit |
| Chip-Pins | 91 Referenztransaktionen (36 062 Byte) über die asynchrone Schnittstelle | Bit für Bit |
| Formal | Handshake und Streaming-Protokoll, k-Induktion (SymbiYosys) | bewiesen |
| Mit IHP-Pads | dieselbe Suite über die Zellmodelle sg13g2_io | Bit für Bit |
| Lint | Verilator -Wall, Yosys 0.33 und 0.69, Icarus | 0 Warnungen |

*Tabelle 3.* Details: [`spec/05_verification.md`](../spec/05_verification.md).

### 3.5 Physische Implementierung

IHP SG13G2 (130 nm), offener Flow LibreLane [13]: Die mit 2,0×2,0 mm,
31 Bond-Pads, Seal Ring, Metall-Fill, Versorgung 1,2 V / 3,3 V, Frequenz
50 MHz. Parallel dazu entstanden eine Variante für GF180MCU (wafer.space) und
eine verkleinerte Version mit 32 Werten für Tiny Tapeout.

| Problem | Ursache | Korrektur | Vorher → nachher |
|---|---|---|---|
| Timing im Slow Corner (1,08 V, 125 °C) | schwache Buffer an großen Fan-outs; Reparatur nach geschätzten parasitären Kapazitäten nach der Platzierung; Prüfung nur des Typical Corner | Reparatur nach dem globalen Routing, Prüfung aller Corners, max transition 1,5 ns bei der Platzierung | −2,5 ns → +1,08 ns |
| Überzählige Antennendioden | eine Heuristik fügte an fast jedem Netz eine Diode ein | Heuristik abgeschaltet, Antennen werden nach dem Routing repariert | 47 343 → 0 |
| Hold-Buffer | die Taktunsicherheit von 0,25 ns wurde auf Hold angewendet | eigene Hold-Slack-Marge von 0,10 ns | 7 254 → 731 |
| Speichermangel beim Fill | das IHP-Skript expandiert das Die in flache Geometrie | dieselben Regeln im hierarchischen Modus von KLayout | >12 GB → 2 GB |
| Metal2-Dichte (≥ 25 % im 800-µm-Fenster) | der Kern ist vollständig mit Zellen belegt | Kern 1,04 mm statt 1,2 mm, Fill-Pitch je Layer | 16–22 % → Regel in allen Fenstern erfüllt |

*Tabelle 4.* Die Zellfläche sank von 1,05 auf 0,68 mm², die Kernauslastung
von 73 auf 63 %.

| Sign-off-Prüfung | Werkzeug | Ergebnis |
|---|---|---|
| DRC des Layouts mit Fill, vollständiger IHP-Regelsatz (174 Kategorien) | KLayout | 0 Verletzungen |
| Dichte von Metall und Aktivgebiet | KLayout, IHP-Regeln | 0 Verletzungen |
| Antennen | OpenROAD und KLayout | 0 / 0 |
| LVS | Magic + Netgen | Schaltungen stimmen eindeutig überein |
| Timing: Setup / Hold, Slow Corner | OpenSTA | +1,08 / +0,38 ns |
| Timing: Hold, Fast Corner | OpenSTA | +0,08 ns |
| Flanken und Kapazitäten, alle Corners | OpenSTA | 0 Verletzungen |
| Leistung, Typical Corner | OpenROAD | 4,6 mW |

*Tabelle 4a.* Ergebnis des finalen Laufs. Vollständig:
[`spec/06_physical_design.md`](../spec/06_physical_design.md) und
`tapeout/ihp-sg13g2/SIGNOFF.md`.

Der Abgleich von Layout und Schaltplan
(LVS) schlug zunächst an 13 Netzen fehl: Die Bond-Pads lagen stumpf am
Anschluss des Pads an, und Magic erkannte bei der Extraktion aus den Abstracts
keinen Kontakt. Die Bond-Pads wurden um 1 µm verschoben und überlappen jetzt
das Metall des Pads. Der Referenz-Regelsatz von IHP für KLayout (`sg13g2.lvs`)
akzeptiert die Schaltpläne der I/O-Zellen aus demselben PDK nicht. Deshalb
führen Magic und Netgen den LVS-Sign-off aus: Die I/O-Zellen gehen als
Abstracts ein, jede Verbindung zu ihnen wird geprüft, ihr Inhalt jedoch nicht
(IP der Foundry).

## 4. Ergebnisse: Version 1 gegen Version 2

| Merkmal | Version 1 | Version 2 (NQX-S1) |
|---|---|---|
| Rotationszustand, d = 128 | 1 910 Byte ROM | 12 Byte (3 Register) |
| Rotationsarithmetik | FP32-Multiplizierer (im Modell) | CORDIC mit Shifts und Additionen |
| Schicht L3 | wird unnötig berechnet | wird in Hardware übersprungen |
| RMSE, KV-ähnliche Daten, 4 Bit | 0,316 | 0,146 |
| Rotationsgenauigkeit | float32 | ≤ 3·10⁻⁵ rel., Rücktransformation ≤ 1 LSB |
| Kompression | 4,00× (ohne Header) | 3,66× (70 Byte mit Header) |
| Zellfläche des Kerns | — | 0,60 mm² (Synthese), 0,68 mm² nach der Platzierung |
| Flipflops | — | 5 634, alle mit Reset |
| Beleg der Zahlen | analytisches Modell | RTL-Simulation, STA, DRC |

| Variante (Quantisierer S1, außer der ersten) | Relativer RMSE |
|---|---|
| Version 1, Pipeline NQX-Core | 0,316 |
| Ohne Rotation | 0,161 |
| Goldener Winkel, NQX-S1 | 0,146 |
| Goldener Winkel, Butterfly-Topologie (7 Schichten) | 0,133 |
| Zufällige orthogonale Matrix | 0,122 |
| Randomisierter Hadamard | 0,120 |

*Tabelle 5.* Synthetische, KV-Cache-ähnliche Daten (Kanäle mit Ausreißern),
4 Bit pro Wert. Die ersten drei Zeilen stammen aus
[`spec/04_numerics_results.md`](../spec/04_numerics_results.md) (256
Vektoren), die übrigen aus der Durchmischungsanalyse (400 Vektoren, §5).

## 5. Warum der goldene Winkel der zufälligen Rotation unterliegt

Die Rotation hilft der Quantisierung nur, wenn sich die Energie eines
Ausreißers über viele Koordinaten verteilt. Das Maß: wie viele Ausgangskanäle
einen merklichen Anteil der Energie eines Eingangskanals erhalten.

| Rotation, d = 128 | Ausgangskanäle pro Eingangskanal (Median) |
|---|---|
| Goldener Winkel, benachbarte Paare (Versionen 1 und 2) | 4 |
| Goldener Winkel, Butterfly (Paare mit Abstand 1, 2, 4 … 64) | 58 |
| Zufällige orthogonale Matrix | 127 |
| Randomisierter Hadamard | 128 |

Behält man die goldenen Winkel bei und stellt nur das Paarschema auf Butterfly
um, sinkt der Fehler von 0,146 auf 0,133: Der Rückstand zur zufälligen Rotation
schrumpft um mehr als die Hälfte. Die Schwäche der Versionen 1 und 2 ist also
vor allem topologisch. Das schränkt auch das Hauptargument für den goldenen
Winkel ein: Keinen Speicherbedarf und Determinismus bietet auch die
randomisierte Hadamard-Transformation, die laut Tabelle 5 genauer und in
Hardware einfacher ist (nur Additionen und Subtraktionen). Eine Variante mit
lernbaren Butterfly-Winkeln ist als ButterflyQuant [5] beschrieben.

## 6. Diskussion: ein Chip für die Aufgabe

NQX-S1 ist nicht als universeller Beschleuniger gedacht. Ziel des Projekts ist
ein Template: Der Kompressionsalgorithmus wird an eine konkrete Aufgabe
angepasst (etwa an die Daten des Bildverarbeitungssystems eines Roboters), und
nach dem Template entsteht ein Chip genau für diese Aufgabe. Diesen Ansatz gibt
es in Teilen bereits: Taalas HC1 verdrahtet ein einzelnes Sprachmodell fest im
Silizium [9]; Robomorphic Computing baut einen Beschleuniger aus einem
parametrisierten Template nach dem Aufbau des Roboters [10]; ECON-T am CERN
komprimiert Detektordaten mit einem neuronalen Netz, dessen Gewichte je Zone
einstellbar sind [11]; SpinQuant trainiert die Rotationen für das jeweilige
Modell [3].

Das Bewertungskriterium ändert sich dabei: Entscheidend ist der Gewinn auf den
Daten der konkreten Aufgabe, nicht die Überlegenheit gegenüber der zufälligen
Rotation auf Durchschnittsdaten. Eine Anpassung ist zulässig, wenn die
Aufgabendaten in einen Teil für die Anpassung und einen zurückgehaltenen Teil
(held-out) für die Prüfung geteilt werden und die Parameter erst nach der
Prüfung auf den zurückgehaltenen Daten ins Silizium gehen. Der typische Grund,
warum sich ein Ergebnis nach dem Abschalten der Anpassung verschlechtert:
Anpassung und Prüfung auf denselben Daten.

## 7. Einschränkungen

- Alle Qualitätsschätzungen stammen aus synthetischen Daten; Läufe auf echten
  KV-Caches und Perplexitätsmessungen gab es noch nicht.
- Der Chip ist nicht gefertigt; Frequenz, Leistung und Timing-Slacks sind
  Ergebnisse der statischen Analyse mit den PDK-Bibliotheken.
- Die Geschwindigkeit ist durch den 8-Bit-Bus begrenzt: Es ist ein Testchip
  zur Prüfung der Idee.
- Scan-Ketten gibt es nicht; der gefertigte Chip wird funktional geprüft, mit
  Golden Vectors (Referenzvektoren) über die Pins.
- LVS prüft die I/O-Zellen von IHP als Abstracts: alle Verbindungen zu
  ihnen, aber nicht ihren Inhalt (IP der Foundry).

## 8. Version 3: Plan

1. **Programmierbare Rotationstopologie:** Paarabstand für jede Schicht und
   konstanter Winkel, sodass ein Die sowohl als goldener Butterfly als auch als
   Hadamard-ähnliche Transformation arbeitet.
2. **Parameter-Tuner:** Auswahl von Topologie, Winkeln und Quantisierungsstufen
   anhand der Aufgabendaten mit verpflichtender Prüfung auf zurückgehaltenen
   Daten; am Ausgang steht eine Konfiguration für den Chipgenerator.
3. **Prüfung auf realen Daten** nach einem vorab festgelegten Protokoll.
4. **FPGA-Prototyp** aus demselben RTL.
5. **Fertigung** der ersten Iteration über Tiny Tapeout (GF180), danach
   wafer.space oder IHP.

## 9. Fazit

Version 1 zeigte, dass goldene Winkel eine kompakte und deterministische
Rotation liefern, aber keine genauere. Version 2 machte aus dem Algorithmus
einen verifizierten Testchip und lieferte drei Ergebnisse, die wichtiger sind
als die ursprüngliche Hypothese: Die dritte Schicht des Algorithmus ist die
Identität; der gesamte Rotationszustand reduziert sich auf zwei Konstanten; den
Fehler halbiert der Quantisierer, nicht die Wahl der Winkel. Die
Durchmischungsanalyse zeigte, dass die Schwachstelle das Paarschema ist und
nicht der Winkel. Sie wies die Richtung für die dritte Version: ein
programmierbares Template, das sich anhand der Daten mit getrennter Prüfung an
die Aufgabe anpasst.

---

# Teil II. Rückwärtsanalyse: wo NQX gewinnt und warum

## 10. Umgekehrte Fragestellung

Teil I beantwortete die Frage „Ist der goldene Winkel im Mittel besser?“ und
erhielt eine negative Antwort. Hier lautet die Frage umgekehrt: Welche
Eigenschaften haben offene Projekte, die dieselbe Aufgabe lösen, und an welchem
Punkt des Anforderungsraums entscheiden die Eigenschaften von NQX den Ausgang?
Jedes Szenario ist eine Messung, keine Annahme. Code:
[`research/reverse_study.py`](../research/reverse_study.py),
Ergebnisse: [`research/reverse_study_results.md`](../research/reverse_study_results.md).

### 10.1 Was offene Projekte tun

| Projekt | Was es mit KV macht | Rotation | Quantisierer | Hardwarekosten |
|---|---|---|---|---|
| KIVI [4] | K pro Kanal, V pro Token, 2–4 Bit | keine | uniform, asymmetrisch, Token-Gruppen | Puffer der letzten Token in voller Genauigkeit |
| KVQuant [14] | K pro Kanal **vor RoPE**, ungleichmäßiges Gitter | keine | NUQ, kalibrierte Skalen | RoPE nach der Dequantisierung (Multiplikationen) |
| PolarQuant [15] | Key als RoPE-Paare in Polarform | keine | Radius und Winkel des Paares | Attention über eine Tabelle q·k nach Codes |
| TurboQuant [1] | zufällige Rotation + polarer Code + QJL | zufällige orthogonale | skalar + 1 Bit | d² Zustand, d² Multiplikationen |
| QuaRot [2] | randomisierter Hadamard | Walsh–Hadamard | int4 | d log d Additionen, nur 2ᵏ |
| SpinQuant [3] | trainierte Rotation | trainierte Matrix | int4 | d² pro Schicht, Training |
| ButterflyQuant [5], HARP [16] | lernbarer Butterfly aus Givens-Rotationen | strukturiert | int2–4 | n log n / 2 Parameter; HARP unterstützt Nicht-2ᵏ |
| llama.cpp q4_0/q8_0 | Blöcke zu 32 Werten | keine | symmetrisch + fp16-Skala | 4,5 Bit, billige Dequantisierung |
| NVFP4 (Blackwell) | Blöcke zu 16, E2M1 + FP8-Skala | keine | Gleitkommaformat | Hardwareunterstützung in der GPU |

Offenes Silizium für KV-Cache-Kompression gibt es nicht. Am nächsten kommen ein
Block für „lokale Rotation“ in ISSCC 2026 31.1 und simulierte Beschleuniger
(siehe `spec/09_landscape.md`). Die Übersicht [17] (200 Arbeiten,
43 Transformationsverfahren) formuliert ein allgemeines Prinzip: Quantisierung
mit gemeinsamer Skala pro Gruppe profitiert von einer „Einebnung“ der Energie
innerhalb der Gruppe, Kodierung mit flexibler Bitverteilung dagegen von ihrer
Konzentration.

### 10.2 Worin sich NQX von allen unterscheidet

1. **Rotationszustand** — 12 Byte (Phaseninkremente) gegenüber 32 KB bei der
   zufälligen Matrix und 16 Byte Vorzeichen beim randomisierten Hadamard.
2. **Arithmetik** — CORDIC mit Shifts und Additionen, ohne Multiplizierer.
3. **Dimension** — jede gerade, ohne die Anforderung 2ᵏ.
4. **Determinismus** — ganzzahliger Datenpfad, das Ergebnis ist auf jedem
   Gerät gleich.
5. **Paare in Polarform** — dieselbe Datenform, in der RoPE die Paare des Keys
   dreht.
6. **Programmierbarkeit** — Phaseninkremente in Registern: Eine
   Transformation lässt sich durch eine andere ersetzen, ohne das Die neu zu
   fertigen.

### 10.3 KV-Cache-Emulator

Gewichte offener Modelle lassen sich von der Build-Maschine nicht
herunterladen (Hugging Face ist durch die Netzwerkrichtlinie der Umgebung
gesperrt). Deshalb sind die Daten anhand veröffentlichter Eigenschaften
emuliert: RoPE mit Paaren (i, i + d/2) und Basis 500 000; in Keys und Queries
sitzen große Werte in einigen niederfrequenten RoPE-Paaren, in einer der beiden
Dimensionen des Paares [4, 15, 18]; Values haben keine Kanalausreißer [4].
Metriken: relativer RMSE von Keys und Values und relativer Fehler der
Attention-Ausgabe softmax(q·k/√d)·V für die letzten Queries, also das, was das
Modell tatsächlich erhält. Verfahren mit Kalibrierung nehmen ihre Konstanten
aus einer anderen Sequenz desselben Heads. Die Bits pro Wert berücksichtigen
alle Skalen und Header.

## 11. Ergebnisse

### 11.1 Keys: alle Verfahren bei ~4,25 Bit

| Verfahren (Keys; Values bei allen INT4 pro Token) | Bit/Wert | ohne Token-Puffer | RMSE Keys | Attention-Fehler |
|---|---|---|---|---|
| KIVI-4, Gruppen zu 128 Token | 4,25 | nein | 0,060 | **0,233** |
| **NQX-RN**: Keys vor RoPE, polar über RoPE-Paare, RoPE als Winkeladdition | 4,25 | nein | 0,073 | 0,336 |
| TurboQuant-ähnlich: zufällig + S1 | 4,25 | ja | 0,122 | 0,338 |
| Abgestimmter goldener Butterfly + S1 (7 Register) | 4,25 | ja | 0,128 | 0,347 |
| Goldener Butterfly + S1 | 4,25 | ja | 0,131 | 0,356 |
| QuaRot-ähnlich: Hadamard + S1 | 4,25 | ja | 0,120 | 0,372 |
| PolarQuant-ähnlich: S1 auf RoPE-Paaren | 4,25 | ja | 0,164 | 0,386 |
| KVQuant-ähnlich, kalibriert | 4,00 | ja | 0,082 | 0,409 |
| KIVI mit kalibrierten Bereichen | 4,00 | ja | 0,084 | 0,421 |
| NQX-RN mit kalibrierten Konstanten | 4,00 | ja | 0,081 | 0,434 |
| NQX-S1 wie gefertigt (benachbarte Paare) | 4,25 | ja | 0,148 | 0,463 |
| S1 ohne Rotation | 4,25 | ja | 0,161 | 0,479 |
| llama.cpp q4_0 | 4,50 | ja | 0,154 | 0,496 |
| INT4 pro Token | 4,25 | ja | 0,200 | 0,502 |

*Tabelle 6.* d = 128, 16 Heads, 1024 Token, 64 letzte Queries.

### 11.2 Signifikanzprüfung

Die Unterschiede im Attention-Fehler zwischen dem goldenen Butterfly, seiner
abgestimmten Variante, Hadamard und zufälliger Rotation sind kleiner als ein
bis anderthalb Standardfehler und wechseln das Vorzeichen von einer
Head-Population zur anderen (drei unabhängige Populationen zu je 16 Heads).
**Diese vier Rotationen sind statistisch nicht unterscheidbar.** NQX-S1 in der
gefertigten Topologie ist durchweg schlechter als Hadamard: um 0,09–0,20,
2,1 bis 2,8 Standardfehler in jeder der drei Populationen
(`research/reverse_study_results.md`, Abschnitt 5).

### 11.3 Head-Dimension

| d | ohne Rotation | golden, wie in NQX-S1 | goldener Butterfly | zufällig | Hadamard (blockweise bei d ≠ 2ᵏ) | Hadamard mit Zero-Padding |
|---|---|---|---|---|---|---|
| 64 | 0,156 | 0,135 | 0,126 | 0,120 | 0,119 | — |
| 80 | 0,157 | 0,142 | 0,137 | 0,121 | 0,120 | 0,095 (+60 % Bit) |
| 96 | 0,156 | 0,139 | 0,129 | 0,121 | 0,123 | 0,104 (+33 % Bit) |
| 128 | 0,163 | 0,150 | 0,131 | 0,121 | 0,120 | — |

*Tabelle 7.* RMSE der Keys. Der blockweise Hadamard (64 + 16, 64 + 32) kommt
mit Nicht-2ᵏ-Dimensionen nicht schlechter zurecht als die zufällige Rotation.
Einen Qualitätsvorteil hat NQX hier nicht.

### 11.4 Determinismus über Plattformen hinweg

Eine zufällige Rotation in fp32 mit unterschiedlicher Summationsreihenfolge
ergibt eine Differenz von 3·10⁻⁶ und ändert keinen einzigen 4-Bit-Code. Rechnet
aber eine Plattform in fp32 und eine andere mit Ein- und Ausgängen in bf16 (der
übliche Modus von Beschleunigern), **erhalten 2,1 % der Paare einen anderen
Code und 73 % der Vektoren mindestens einen abweichenden Code.** Der
ganzzahlige CORDIC von NQX-S1 (und ein ganzzahliger Hadamard) liefern
konstruktionsbedingt 0 Abweichungen.

### 11.5 Abstimmung des „Templates“ auf die Aufgabe

| Abstimmung der 7 Butterfly-Register | auf den Abstimmungsdaten | auf den zurückgehaltenen Daten |
|---|---|---|
| nach RMSE der Keys; zurückgehalten: neue Token derselben Heads | 0,1358 → 0,1253 (−7,7 %) | 0,1358 → 0,1253 (−7,7 %) |
| nach Attention-Fehler; zurückgehalten: **andere** Heads | 0,425 → 0,361 (−15 %) | 0,350 → 0,354 (+1 %, im Rauschen) |
| nach Attention-Fehler, **eigene Abstimmung für jeden Head**; zurückgehalten: neue Token | 0,333 → 0,212 (−36 %) | 0,391 → 0,395 (+1 %; Hadamard 0,384, zufällig 0,361) |

*Tabelle 8.* Die zweite und dritte Zeile reproduzieren in der Emulation, was
in früheren Experimenten des Projekts beobachtet wurde: Das abgestimmte
Ergebnis sieht auf den Abstimmungsdaten stark aus und verschwindet auf anderen
Daten, selbst auf neuen Token desselben Heads (12 Heads: Gewinn gegenüber dem
ursprünglichen Butterfly in 6 von 12, Differenz +0,004 ± 0,026). Der
Attention-Fehler auf 32 Queries ist eine verrauschte Zielfunktion: 7 Register
passen sich in 140 Schritten an das Rauschen der konkreten Stichprobe an.
Übertragbar ist nur eine Abstimmung nach robuster Statistik (RMSE der Keys über
die gesamte Verteilung, erste Zeile).

### 11.6 Hardwarekosten der Rotation (d = 128)

| Rotation | Zustand | Dimension | Multiplizierer | Operationen pro Vektor | Bitgenauigkeit |
|---|---|---|---|---|---|
| NQX-S1 (CORDIC, 2 Schichten) | 12 B | jede gerade | 0 | 6 858 Additionen | ja |
| Goldener Butterfly, programmierbar | 28 B (7 Register) | jede | 0 | 24 192 Additionen | ja |
| Goldener Butterfly, Winkel in der Maske | 0 | jede | 0 | ~5 400 Additionen | ja |
| Randomisierter Hadamard (FWHT) | 16 B | 2ᵏ oder Blöcke | 0 | 896 Additionen | ja, in Ganzzahlen |
| Zufällig (TurboQuant) | 32 KB | jede | 16 384 | 16 384 MAC | nein |
| Trainiert (SpinQuant) | 32 KB pro Schicht | jede | 16 384 | 16 384 MAC | nein |
| RoPE als Winkeladdition (NQX-RN) | Phasenakkumulator pro Paar | jede gerade | 0 | 64 Additionen statt 256 Multiplikationen + 128 Additionen | ja |

## 12. Wo NQX gewinnt und wo nicht

**Szenario A. Deterministische KV-Kompression ohne Multiplizierer am Netzrand**
(Roboter, eingebettetes oder zertifizierbares System). Gefordert: kein
Token-Puffer, keine Multiplizierer, keine gespeicherten Matrizen, gleiches
Ergebnis auf jedem Gerät. Der goldene Butterfly liefert denselben
Attention-Fehler wie die besten Streaming-Rotationen (Tabelle 6, Abschn. 11.2).
Gegenüber TurboQuant und SpinQuant kommt er mit einem tausendfach kleineren
Zustand aus, nutzt keine Multiplizierer und ist bitgenau, wo eine
Float-Rotation beim Übergang fp32 → bf16 die Codes von 73 % der Vektoren
ändert. **Hier gewinnt NQX gegen Float-Rotationen auf allen Achsen bei gleicher
Qualität.** Gegen Hadamard steht es bei Qualität und Determinismus
unentschieden; Hadamard braucht weniger Additionen. Es bleiben die
Programmierbarkeit (Phasenregister) und der gemeinsame CORDIC für Rotation und
polaren Code.

**Szenario B. Keys in der für RoPE „nativen“ Form.** Speichert man die Keys vor
RoPE in Polarform über RoPE-Paare, ist eine Misch-Rotation überhaupt nicht
nötig, und die Positionskodierung wird zur Winkeladdition (64 Additionen statt
256 Multiplikationen und 128 Additionen pro Key). So passt die tabellenbasierte
Attention-Berechnung (wie in PolarQuant [15]) auf denselben NQX-Datenpfad. In
der Emulation erreichte diese Variante ganz ohne Rotation den Attention-Fehler
der besten Verfahren mit Rotation (0,336 gegenüber 0,338 bei der zufälligen),
bei halb so großem RMSE der Keys. Einschränkung: Diese Variante kodierte in
Gruppen zu 128 Token, also mit Puffer, während die Rotationen streamend
arbeiten. Ein fairer Vergleich und ein Streaming-Codec folgen in Teil III. Dies
ist das einzige Szenario, in dem der Aufbau von NQX (Paare, Polarform,
Phasenakkumulatoren) kein Kompromiss ist, sondern genau zur Datenstruktur
passt.

**Wo NQX verliert.** Kann man einen Puffer von 128 Token in voller Genauigkeit
halten, ist die komponentenweise Quantisierung der Keys pro Kanal (KIVI) am
genauesten (0,233). Bei der Zahl der Additionen ist Hadamard günstiger als
jeder Butterfly auf CORDIC-Basis. Die Topologie von NQX-S1 wie gefertigt
(benachbarte Paare) verliert in allen Szenarien. Die Abstimmung des „Templates“
nach der Endmetrik (Attention-Fehler) überträgt sich weder auf andere Heads
noch auf neue Token desselben Heads.

## 13. Folgerungen für Version 3

1. Die Topologie benachbarter Paare durch Butterfly ersetzen: Die Qualität
   zieht mit Hadamard und zufälliger Rotation gleich, bei denselben
   0 Multiplizierern.
2. Einen Paarmodus (i, i + d/2) und einen Pfad für Keys vor RoPE hinzufügen:
   polarer Code über RoPE-Paare, RoPE per Phasenaddition, Konstanten pro Paar
   aus der Kalibrierung.
3. Die Template-Parameter nach robuster Datenstatistik (RMSE der Keys,
   Verteilung über die Kanäle) auf einer großen Stichprobe abstimmen, nicht
   nach einer verrauschten Endmetrik. Eine Abstimmung nur übernehmen, wenn sie
   auf den zurückgehaltenen Daten statistisch signifikant gewinnt; sonst die
   ursprünglichen Winkel beibehalten.
4. Die wichtigste Eigenschaft für den Markt ist nicht die Qualität, sondern
   Bitgenauigkeit ohne Multiplizierer und ohne Speicher für die Rotation:
   Robotik, eingebettete und zertifizierbare Systeme.
5. Alles an echten KV-Caches prüfen (Llama, Qwen, Phi-3 mit d = 96), sobald
   Zugriff auf die Gewichte besteht.

**Einschränkung von Teil II.** Alle Zahlen stammen aus dem Emulator, nicht aus
echten Modellen. Der Emulator reproduziert bekannte Eigenschaften von KV,
ersetzt aber nicht die Prüfung auf realen Daten und die Messung der
Perplexität.

# Teil III. NQX-RN: Keys vor RoPE in Polarform

## 14. Die Idee und ihr Platz unter bekannten Arbeiten

Teil II fand einen einzigen Arbeitspunkt, an dem der Aufbau von NQX keine
zufällige Rotation nachbildet, sondern mit der Datenstruktur übereinstimmt.
Der Key wird **vor RoPE** gespeichert, und jedes RoPE-Paar (i, i + d/2) wird
als Radius und Winkel abgelegt. RoPE dreht das Paar um den Winkel m·ωᵢ. Daher
ist die Positionskodierung in Polarform eine Winkeladdition, und der
Attention-Score lässt sich direkt aus den Codes berechnen:

  q·k = Σᵢ ρᵢ · rᵢ · cos(φᵢ − θᵢ + (n − m)·ωᵢ),

wobei (ρᵢ, φᵢ) das Query-Paar vor RoPE ist, (rᵢ, θᵢ) das Key-Paar und n und m
die Positionen von Query und Key. Diese Form stützt sich auf vier
Eigenschaften:

1. **Die Statistik der Keys vor RoPE ist über die Kanäle stationär** [14]:
   Jeder Kanal hat eine stabile Skala und einen stabilen Mittelwert. Die
   Konstanten des Quantisierers lassen sich einmal pro Head kalibrieren.
2. **Massive Werte sitzen in niederfrequenten RoPE-Paaren**, meist in einer
   der beiden Koordinaten [15, 18]. In Polarform ist ein solches Paar ein fast
   konstanter Radius und ein schmaler Winkelbogen; es lässt sich billig
   kodieren.
3. **Die Energie des Query-Paares |qᵢ|² ändert sich unter RoPE nicht.** Daher
   ist das Gewicht des Paares bei der Bitverteilung für jeden relativen
   Versatz (n − m) exakt. Das Gewicht einer einzelnen Koordinate hat diese
   Eigenschaft nicht: RoPE mischt die beiden Koordinaten des Paares.
4. **Die Position wird zur ganzen Zahl.** Die Phase m·ωᵢ wird als 32-Bit-Wort
   m·Wᵢ mod 2³² gespeichert, Wᵢ = round(ωᵢ·2³²/2π). Das ist exakte modulare
   Arithmetik: kein cos und sin eines großen Arguments in Float.

| Arbeit | Wo der Key quantisiert wird | Form | Positionen | Bitverteilung |
|---|---|---|---|---|
| KVQuant [14] | vor RoPE | kartesisch, pro Kanal | RoPE wird nach der Dequantisierung mit Multiplikationen neu berechnet | keine |
| PolarQuant, Wu et al. [15] | nach RoPE | polar über Paare | bereits im Key enthalten | keine |
| PolarQuant, Han et al. [19] | nach RoPE, nach zufälliger Rotation | rekursiv polar | bereits im Key enthalten | keine |
| Block-GTQ [20] | nach RoPE | kartesisch, TurboQuant-MSE über RoPE-Blöcke | bereits im Key enthalten | über RoPE-Blöcke, nach Q/K-Energie |
| RoPE-ausgerichtete Rotationen [21] | Rotation kommutiert mit RoPE | kartesisch, Rotation innerhalb jedes Paares | ändern sich nicht | keine |
| **NQX-RN** | **vor RoPE** | **polar über RoPE-Paare** | **ganzzahlige Phase, Winkeladdition** | **Radius und Winkel jedes Paares, nach Query-Energie** |

*Tabelle 9.* Die Autoren von PolarQuant [15] stellen ihr Verfahren
ausdrücklich der Quantisierung vor RoPE gegenüber: Bei KVQuant müssen die
Positionen in jedem Dekodierschritt neu berechnet werden. In Polarform
schrumpft diese Neuberechnung auf eine einzige
ganzzahlige Phasenaddition. So lassen sich beide Vorteile zugleich nutzen: die
stationäre Statistik vor RoPE und der Wegfall der Neuberechnung. In den von
uns gefundenen Arbeiten gibt es diese Kombination nicht. Das ist eine Aussage
über die gesichtete Literatur, kein Neuheitsbeweis. Die Arbeit [21] zeigte,
dass reine Rotationen innerhalb der RoPE-Paare dem vollen Hadamard
unterliegen; Tabelle 12 unten stimmt damit überein: Die lokale, paarweise Form
allein bringt keinen Gewinn.

## 15. Codec

- **Token-Skala:** der größte Radius unter den Paaren des Keys, 16 Bit. Das
  ist derselbe rmax-Header, den NQX-S1 schreibt. Ein einzelner Komparator
  findet ihn. Mittelwert und RMS der Radien sind in der Emulation schlechter
  (separater Lauf, 16 Heads, 3 Bit: 0,346 und 0,301 gegenüber 0,282).
- **Radius des Paares:** rᵢ/s wird gleichmäßig auf einem kalibrierten
  Intervall quantisiert (Perzentile 0,1 und 99,9 %).
- **Winkel des Paares:** Die Abweichung vom kalibrierten Bogenmittelpunkt cᵢ
  wird auf dem Bogen ±wᵢ quantisiert (Perzentil 99,5 %). Ist der Bogen breiter
  als 0,8π, ist der Code zirkulär.
- **Bits des Paares:** Ein Greedy-Algorithmus verteilt das Gesamtbudget (2B Bit
  pro Paar bei B Bit pro Wert) zwischen Radius und Winkel jedes Paares. Das
  Gewicht des Paares ist die mittlere Query-Energie E|qᵢ|², der Fehler ist der
  mittlere quadratische Fehler des Paares auf der Kalibrierungssequenz.
- **Kalibrierung:** eine Sequenz desselben Heads, getrennt von der
  Prüfsequenz. Konstanten pro Paar: Radiusgrenzen, Mittelpunkt und Breite des
  Bogens zu je 16 Bit plus zwei Bitbreiten zu je 3 Bit, 70 Bit pro Paar,
  560 Byte pro Head bei d = 128.

Speicherbedarf eines Keys bei 3 Bit pro Wert: 64 Paare × 6 Bit + 16 Bit Skala
= 50 Byte statt 256 Byte in fp16.

**Korrektur zu Teil II.** Die NQX-RN-Variante aus Teil II (0,336) quantisierte
in Gruppen zu 128 Token. Sie braucht einen Token-Puffer wie KIVI, daher war der
Vergleich mit Streaming-Rotationen unfair. Ihre Streaming-Variante mit
statischer Kalibrierung verlor (0,434). Der Codec dieses Abschnitts arbeitet
streamend.

## 16. Ergebnisse

Emulator und Metrik sind dieselben wie in Teil II: d = 128, 1024 Token, Fehler
der Attention-Ausgabe für die 64 letzten Queries, Values bei allen Verfahren
INT4 pro Token. Drei unabhängige Populationen zu je 16 Heads, paarweise
Differenzen. Starker Konkurrent mit Rotation: zufällige Rotation oder
Hadamard-Rotation, danach ein skalarer Lloyd–Max-Quantisierer pro Koordinate,
16-Bit-Norm (Schema TurboQuant-MSE [1]).

| Key-Verfahren | Bit pro Wert | Streaming | Kalibrierung | Attention-Fehler |
|---|---|---|---|---|
| NQX-RN, 4 Bit | 4,125 | ja | ja | **0,198** |
| Kartesisch vor RoPE (wie KVQuant), 4 Bit | 4,125 | ja | ja | 0,212 |
| KIVI-4, Puffer 128 Token | 4,25 | nein | nein | 0,251 |
| **NQX-RN, 3 Bit** | **3,125** | ja | ja | **0,263** |
| Hadamard + Lloyd–Max, 4 Bit | 4,125 | ja | nein | 0,297 |
| Zufällige Rotation + Lloyd–Max, 4 Bit | 4,125 | ja | nein | 0,316 |
| Kartesisch vor RoPE, 3 Bit | 3,125 | ja | ja | 0,334 |
| Zufällige Rotation + S1 (Teil II) | 4,25 | ja | nein | 0,357 |
| Goldener Butterfly + S1 (Teil II) | 4,25 | ja | nein | 0,361 |
| NQX-RN aus Teil II, Gruppen zu 128 Token | 4,25 | nein | nein | 0,386 |
| **NQX-RN, 2 Bit** | **2,125** | ja | ja | **0,404** |
| Hadamard + Lloyd–Max, 3 Bit | 3,125 | ja | nein | 0,478 |
| Kartesisch vor RoPE, 2 Bit, Bits pro Paar | 2,125 | ja | ja | 0,489 |
| Hadamard + Lloyd–Max, 2 Bit | 2,125 | ja | nein | 0,824 |

*Tabelle 10.* 48 Heads. Vollständig: `research/nqx_rn_results.md`.

| Paarweise Differenz (kleiner als null — NQX-RN besser) | Mittelwert ± Std.-Fehler | NQX-RN besser |
|---|---|---|
| NQX-RN 3 Bit − Hadamard + Lloyd–Max 4 Bit | −0,033 ± 0,014 | 33 von 48 |
| NQX-RN 3 Bit − zufällige Rotation + Lloyd–Max 4 Bit | −0,052 ± 0,016 | 36 von 48 |
| NQX-RN 3 Bit − zufällige Rotation + S1, 4,25 Bit | −0,093 ± 0,019 | 38 von 48 |
| NQX-RN 3 Bit − KIVI-4 mit Puffer | +0,012 ± 0,019 | 17 von 48 (unentschieden) |
| NQX-RN 3 Bit − bestes kartesisches vor RoPE, 3 Bit | −0,070 ± 0,014 | 40 von 48 |
| NQX-RN 2 Bit − bestes kartesisches vor RoPE, 2 Bit | −0,085 ± 0,015 | 43 von 48 |
| NQX-RN 4 Bit − bestes kartesisches vor RoPE, 4 Bit | −0,014 ± 0,009 | 36 von 48 (grenzwertig) |

*Tabelle 11.*

**Hauptergebnis der Emulation: NQX-RN braucht etwa ein Bit pro Wert weniger
als die beste Rotation, bei gleichem oder kleinerem Fehler.** Bei 3,125 Bit
ist es genauer als Hadamard und zufällige Rotation bei 4,125 Bit. Bei
2,125 Bit ist es genauer als Hadamard bei 3,125. Gegen KIVI steht es bei 3 Bit
gegenüber 4,25 unentschieden, doch KIVI braucht einen Puffer von 128 Token in
voller Genauigkeit.

**Woher der Gewinn kommt.** Zerlegung nach Komponenten bei 3 Bit:

| Variante | Attention-Fehler |
|---|---|
| kartesisch vor RoPE, gleiche Bits | 0,334 |
| kartesisch vor RoPE, Bits pro Paar mit Gewicht nach Query-Energie (wie Block-GTQ) | 0,369 |
| polar, 3 + 3 Bit für jedes Paar | 0,347 |
| polar, Bits nach Key-Fehler ohne Query-Gewicht | 0,354 |
| polar, Bits mit Gewicht nach Query-Energie (NQX-RN) | **0,263** |

*Tabelle 12.* Die Polarform allein ist nicht besser als die kartesische (0,347
gegenüber 0,334). Den Gewinn bringt die Kombination aus Polarform und
Bitverteilung nach Query-Energie. In kartesischer Form half dieselbe
Verteilung nicht, auch nicht pro Paar (0,369). Unsere Erklärung: Die Polarform
trennt Komponenten, die unterschiedlich wichtig sind: den Radius, der bei
massiven Paaren fast konstant ist, und den Winkel, der bei Rauschpaaren
gleichverteilt ist. Das Gewicht ist dabei bei jedem Positionsversatz exakt.
Das ist eine Erklärung, kein bewiesener Mechanismus.

## 17. Wo NQX-RN nicht mehr gewinnt

Der Emulator setzt Eigenschaften voraus, die in der Literatur für echte
Modelle beschrieben sind: stationäre Kanäle vor RoPE und massive Werte in
niederfrequenten Paaren. Die Prüfung bricht nacheinander jede Annahme
(16 Heads pro Szenario):

| Szenario | Hadamard, 4 Bit | kartesisch, 3 Bit | NQX-RN, 3 Bit | Hadamard, 3 Bit | Ergebnis für NQX-RN bei 3 Bit |
|---|---|---|---|---|---|
| Basis | 0,307 | 0,318 | **0,260** | 0,504 | am besten |
| ohne massive Werte | **0,203** | 0,472 | 0,389 | 0,334 | verliert gegen Rotation |
| massiver Wert in beiden Koordinaten des Paares | 0,299 | 0,333 | **0,268** | 0,514 | am besten |
| massive Größe, Richtung wechselt von Token zu Token | 0,214 | 0,450 | **0,177** | 0,353 | am besten; kartesisch bricht zusammen |
| Statistik gegenüber der Kalibrierung um 20 % verschoben | 0,300 | 0,452 | 0,353 | 0,472 | unentschieden gegen Hadamard bei 4 Bit |
| Statistik um 40 % verschoben | **0,285** | 0,595 | 0,539 | 0,501 | verliert gegen Hadamard bei 4 Bit, unentschieden bei 3 |
| RoPE, Basis 10 000 | 0,330 | 0,348 | **0,266** | 0,497 | am besten |
| d = 96 | 0,377 | 0,329 | **0,254** | 0,579 | am besten |
| d = 64 | 0,349 | 0,368 | **0,251** | 0,739 | am besten |

*Tabelle 13.* Folgerungen:

- **Es braucht Struktur in den Daten.** Ohne massive Werte sind die Keys
  nahezu gaußverteilt, die Kalibrierung bringt nichts, und die Rotation
  gewinnt. Der gesamte Gewinn von NQX-RN beruht auf einer Eigenschaft, die an
  echten Modellen zu bestätigen ist.
- **Die Polarform ist robust gegen eine Rotation der Ausreißer innerhalb des
  Paares.** Ändert die massive Größe ihre Richtung von Token zu Token, bricht
  der kartesische Code vor RoPE zusammen (0,450), während der polare am
  besten arbeitet (0,177): Der Radius ist konstant. Das ist die Beobachtung
  von PolarQuant [15], übertragen auf Keys vor RoPE.
- **Die Kalibrierung ist das Hauptrisiko.** Bei einer Verschiebung der
  Statistik um 20 % verschwindet der Vorsprung gegenüber Hadamard bei 4 Bit, bei 40 % verliert NQX-RN gegen ihn. Das ist dieselbe Falle wie bei der Abstimmung
  der Winkel in Teil II. Nötig sind eine Rekalibrierung im laufenden Betrieb
  (gleitende Perzentile im Kodierpfad) oder ein Rückfall auf die
  Hadamard-Rotation, wenn der Kodierfehler steigt.

## 18. Hardware

**Positionen.** Fehler der Attention-Ausgabe bei exakten Keys, RoPE auf
verschiedene Weise berechnet, an den Positionen 0 … 2²⁰:

| RoPE-Basis, Position | fp32 | Winkel fp32, cos/sin in bf16 | Winkel in bf16 | ganzzahlige Phase 16 Bit | 24 Bit | 32 Bit |
|---|---|---|---|---|---|---|
| 500 000, 0 | 1,6·10⁻⁶ | 1,1·10⁻³ | 0,14 | 5,2·10⁻³ | 4,2·10⁻⁵ | 2,5·10⁻⁵ |
| 500 000, 131 072 | 4,0·10⁻⁴ | 7,0·10⁻³ | 0,52 | 5,2·10⁻³ | 8,3·10⁻⁵ | 7,4·10⁻⁵ |
| 500 000, 1 048 576 | 3,8·10⁻³ | 8,8·10⁻³ | 0,52 | 5,2·10⁻³ | 6,7·10⁻⁵ | 6,8·10⁻⁵ |
| 10 000, 1 048 576 | 5,8·10⁻³ | 1,4·10⁻² | 0,64 | 7,9·10⁻³ | 1,2·10⁻⁴ | 7,8·10⁻⁵ |

*Tabelle 14.* Der Fehler von Float-RoPE wächst mit der Position: Die absolute
Genauigkeit des Winkels m·ω sinkt. Ein Winkel in bf16 macht RoPE vollständig
unbrauchbar; das ist ein bekanntes Problem [22]. Der übliche bf16-Modus der
Modelle (Winkel in fp32, cos und sin in bf16) ergibt 1–14·10⁻³. **Die
ganzzahlige Phase hängt konstruktionsbedingt nicht von der Position ab.**
24 Bit reichen für einen Fehler von etwa 10⁻⁴; die Grenze setzen die
16-Bit-Werte von cos und sin.

**Attention-Score direkt aus den Codes.** Der Index der Kosinustabelle sind
die höherwertigen Bits der 32-Bit-Phase φᵢ + (n − m)·Wᵢ − θᵢ. Beim Übergang zum
nächsten Key ändert sich die Phase durch eine einzige Addition. Eine Tabelle
mit 8-Bit-Index (256 Werte zu 12 Bit, 384 Byte) erhöht den Fehler um 0,001
(0,1505 gegenüber 0,1494 bei Float-Berechnung). Eine 6-Bit-Tabelle erhöht ihn
um 0,009.

| Key-Format | Arbeit pro Key und Paar beim Attention-Scan | Multiplikationen |
|---|---|---|
| vor RoPE, kartesisch (KVQuant) | Dequantisierung, RoPE, Skalarprodukt | 8 |
| nach RoPE, kartesisch (KIVI, Rotationen) | Dequantisierung, Skalarprodukt | 4 |
| nach RoPE, polar, Tabelle pro Query (PolarQuant) | 1 Tabellenzugriff, 1 Addition | 0 (aber 64 × 2⁶ Multiplikationen für den Aufbau der Tabelle pro Query) |
| **vor RoPE, polar, über den Winkel (NQX-RN)** | 2 Additionen, 2 Tabellenzugriffe, 1 Multiplikation, 1 Addition | **1** |

*Tabelle 15.* Bei der Zahl der Operationen pro Key ist PolarQuant günstiger
als NQX-RN. Der Vorteil von NQX-RN liegt nicht in den Operationen, sondern in
der Statistik (etwa ein Bit pro Wert) und in den ganzzahligen Positionen.

**Was NQX-S1 bereits hat:**
- CORDIC im Vektormodus (Paar → Radius und Winkel) und im Rotationsmodus;
- einen polaren Quantisierer mit 3-Bit-Codes für Radius und Winkel;
- den rmax-Header pro Vektor, genau die Skala von NQX-RN;
- einen 32-Bit-Phasenakkumulator mit programmierbaren Inkrementen.

**Was fehlt:**
- Paare (i, i + d/2): Die dritte Schicht hat den Abstand 32, aber der Host kann
  die Koordinaten umordnen;
- Kalibrierungskonstanten pro Paar (560 Byte);
- unterschiedliche Codebreiten für verschiedene Paare;
- 64 Phasenwörter Wᵢ statt drei Registern;
- ein Block für die Scores: Kosinustabelle, Radiustabellen pro Query,
  Multiplizierer und Addierer.

Der gefertigte Chip taugt als Teilprüfstand (Kodierung des Keys in Polarform),
implementiert NQX-RN aber nicht vollständig.

## 19. Prüfung an echten Modellen

Ein Versuchsplan, der die wichtigste Einschränkung schließt:

1. **Modelle:** Llama-3.2-1B (d = 64, Basis 500 000), Qwen2.5-1.5B (d = 128),
   Phi-3-mini (d = 96, Basis 10 000), Llama-3.1-8B.
2. **Daten vor RoPE:** Ausgänge von `k_proj` und `q_proj`. In den
   Implementierungen von Hugging Face wird RoPE nach der Projektion angewendet,
   daher genügt es, die Ausgänge der Schicht abzugreifen.
3. **Kalibrierung** auf 512 Token eines Korpus (C4), Prüfung auf einem anderen
   (WikiText-2), danach eine Domänenverschiebung (Kalibrierung auf Text, Prüfung
   auf Code): eine direkte Prüfung des Risikos aus Abschnitt 17.
4. **Metriken:** Perplexität mit quantisierten Keys (Values bei allen INT4),
   Fehler der Attention-Ausgabe pro Schicht, eine Auswahl von
   LongBench-Aufgaben für langen Kontext.
5. **Konkurrenten:** KIVI, KVQuant-ähnlicher Code vor RoPE, PolarQuant,
   Hadamard und zufällige Rotation mit Lloyd–Max, Block-GTQ, falls der Code
   verfügbar ist.
6. **Erfolgskriterien:**
   - NQX-RN ist bei 3,125 Bit in der Perplexität nicht schlechter als Hadamard
     bei 4,125;
   - bei Domänenverschiebung ist der Verlust nicht größer als beim
     KVQuant-ähnlichen Code;
   - massive Werte sitzen tatsächlich in niederfrequenten Paaren, wie der
     Emulator annimmt.

Das sind einige Stunden auf einer GPU. Aus der Build-Umgebung sind die
Modellgewichte nicht zugänglich, daher wurde das Experiment nicht durchgeführt.

## 20. Folgerungen aus Teil III

NQX-RN ist weder eine Rotation noch die Kopie eines bekannten Verfahrens. Es
kombiniert Quantisierung vor RoPE (wie KVQuant), die Polarform über RoPE-Paare
(wie PolarQuant, aber vor RoPE), eine ganzzahlige Phase statt Float-Positionen
und eine Bitverteilung nach der Query-Energie, die sich unter RoPE nicht
ändert. In der Emulation spart es gegenüber den besten Rotationen etwa ein Bit
pro Wert und macht die Positionen bei jeder Kontextlänge exakt. Sein Gewinn
beruht auf zwei Bedingungen: Die Keys enthalten massive Werte in
niederfrequenten Paaren, und die Statistik entfernt sich nicht weit von der
Kalibrierung. Beide Bedingungen lassen sich an echten Modellen in einigen
Stunden prüfen; bis dahin ist es zu früh, NQX-RN im Silizium von Version 3
vorzusehen.


## Literatur

1. Zandieh A. et al. TurboQuant: online vector quantization with near-optimal distortion rate. ICLR 2026. arXiv:2504.19874.
2. Ashkboos S. et al. QuaRot: outlier-free 4-bit inference in rotated LLMs. 2024. arXiv:2404.00456.
3. Liu Z. et al. SpinQuant: LLM quantization with learned rotations. 2024. arXiv:2405.16406.
4. Liu Z. et al. KIVI: a tuning-free asymmetric 2bit quantization for KV cache. 2024. arXiv:2402.02750.
5. ButterflyQuant: lernbare Butterflies aus Givens-Rotationen für die LLM-Quantisierung (Übersicht in [`spec/09_landscape.md`](../spec/09_landscape.md)).
6. Weyl H. Über die Gleichverteilung von Zahlen mod. Eins. Mathematische Annalen 77, 1916.
7. Knuth D. E. The Art of Computer Programming, vol. 3, §6.4.
8. Volder J. E. The CORDIC trigonometric computing technique. IRE Trans. Electronic Computers, 1959.
9. Taalas HC1: hardwired Llama-3.1 8B accelerator. CNX Software, 2026.
10. Neuman S. M. et al. Robomorphic computing. ASPLOS 2021. doi:10.1145/3445814.3446746.
11. Di Guglielmo G. et al. A reconfigurable neural network ASIC for detector front-end data compression at the HL-LHC. 2021. arXiv:2105.01683.
12. IHP Open PDK documentation: filler generation using KLayout.
13. LibreLane: <https://github.com/librelane/librelane>.
14. Hooper C. et al. KVQuant: towards 10 million context length LLM inference with KV cache quantization. NeurIPS 2024.
15. Wu S. et al. PolarQuant: leveraging polar transformation for key cache quantization and decoding acceleration. NeurIPS 2025. arXiv:2502.00527.
16. HARP: Hadamard-preconditioned adaptive rotation processor for extreme LLM quantization. 2026. arXiv:2605.29843.
17. Transforms for LLM quantization: the Great Inversion and format co-design. 2026. arXiv:2608.25188.
18. Jin M. et al. Massive values in self-attention modules are the key to contextual knowledge understanding. ICML 2025. arXiv:2502.01563.
19. Han I. et al. PolarQuant: quantizing KV caches with polar transformation. 2025. arXiv:2502.02617.
20. Liang F., Zhang Y., Jia J. RoPE-aware bit allocation for KV-cache quantization (Block-GTQ). 2026. arXiv:2606.24033.
21. Wang S., Luo Y., Xu N., Cheung C. W. When local variance optimality is not enough: RoPE-aligned Q/K rotations for dynamic 4-bit quantisation. 2026. arXiv:2608.13365.
22. Wang H. et al. When precision meets position: BFloat16 breaks down RoPE in long-context training. 2024. arXiv:2411.13476.

---

Hardware-Implementierung, Verifikation und physisches Design von Version 2
entstanden mit Unterstützung von Claude Code (Anthropic).
