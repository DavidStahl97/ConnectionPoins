# Maussteuerung - 3D Viewport

Die WPF-Anwendung verwendet **HelixToolkit.Wpf** für die 3D-Visualisierung mit umfassender Maussteuerung.

## Maus-Gesten

### 🖱️ Rotation (Linke Maustaste)

**Aktion**: Linke Maustaste gedrückt halten und ziehen

**Funktion**: Dreht das 3D-Modell um seine Achse

**Rotation-Modus**: `Turntable` (Kamera rotiert um das Objekt, ähnlich wie ein Drehteller)

**Eigenschaften**:
- Inertia aktiviert: Modell dreht sich nach Loslassen weiter
- Rotation Sensitivity: 1.0 (Standard)
- Camera Inertia Factor: 0.9 (sanftes Auslaufen)

**Beispiel-Verwendung**:
- Betrachten Sie das Modell von allen Seiten
- Identifizieren Sie Anschlusspunkte an der Rückseite

---

### 🖱️ Pan / Verschieben (Mittlere Maustaste)

**Aktion**: Mittlere Maustaste (Mausrad-Klick) gedrückt halten und ziehen

**Funktion**: Verschiebt die Kamera horizontal/vertikal

**Eigenschaften**:
- Verschiebt das Modell ohne Rotation oder Zoom
- Praktisch für Positionierung im Viewport

**Beispiel-Verwendung**:
- Modell im Viewport zentrieren
- Detail in der Mitte des Bildschirms platzieren

**Alternative für Mäuse ohne Mitteltaste**:
- Verwenden Sie Shift + Linke Maustaste (falls von HelixToolkit unterstützt)

---

### 🖱️ Zoom (Rechte Maustaste oder Mausrad)

#### Variante 1: Mausrad

**Aktion**: Mausrad nach oben/unten scrollen

**Funktion**:
- Nach oben scrollen: Hineinzoomen
- Nach unten scrollen: Herauszoomen

**Eigenschaften**:
- Zoom Sensitivity: 1.0
- Schnellste Methode zum Zoomen

#### Variante 2: Rechte Maustaste

**Aktion**: Rechte Maustaste gedrückt halten und ziehen

**Funktion**:
- Nach oben ziehen: Hineinzoomen
- Nach unten ziehen: Herauszoomen

**Eigenschaften**:
- Präziseres Zoomen als Mausrad
- Bessere Kontrolle bei feinen Anpassungen

**Beispiel-Verwendung**:
- Details von Anschlusspunkten vergrößern
- Überblick über gesamtes Modell erhalten

---

## ViewCube

Der **ViewCube** ist ein interaktives Element in der oberen rechten Ecke des Viewports.

**Funktion**: Schnelle Navigation zu Standardansichten

**Verwendung**:
- Klicken Sie auf eine **Fläche** für Frontalansicht (z.B. "Front", "Back", "Top", "Bottom")
- Klicken Sie auf eine **Kante** für 45°-Ansicht
- Klicken Sie auf eine **Ecke** für isometrische Ansicht

**Verfügbare Ansichten**:
- Front / Back
- Left / Right
- Top / Bottom
- Isometrische Ecken-Ansichten

---

## Koordinatensystem

Das **Koordinatensystem** zeigt die Achsen-Orientierung:

- **Rote Achse**: X-Achse
- **Grüne Achse**: Y-Achse
- **Blaue Achse**: Z-Achse (zeigt "aus dem Bildschirm" nach oben)

**Ein-/Ausblenden**: Checkbox "Koordinatensystem anzeigen" in der Seitenleiste

---

## Kamera-Eigenschaften

### Turntable-Modus

**Beschreibung**: Die Kamera rotiert um das Objekt wie auf einem Drehteller

**Vorteile**:
- Natürliche Rotation
- Vermeidet "Gimbal Lock" (Achsenverriegelung)
- Immer aufrechte Orientierung

**Alternative Modi** (nicht aktiviert):
- `Trackball`: Freie 3D-Rotation
- `Walkthrough`: Ego-Perspektive

### Inertia (Trägheit)

**Aktiviert**: `IsInertiaEnabled="True"`

**Funktion**: Modell dreht sich nach Loslassen der Maustaste weiter

**Eigenschaften**:
- Camera Inertia Factor: 0.9 (90% der ursprünglichen Geschwindigkeit pro Frame)
- Infinite Spin: False (Rotation stoppt nach kurzer Zeit)

**Deaktivieren**: Falls unerwünscht, kann Inertia ausgeschaltet werden

---

## UI-Buttons

### Zoom-Buttons

**🔍+ (Zoom In)**: Vergrößert die Ansicht um Faktor 0.8

**🔍- (Zoom Out)**: Verkleinert die Ansicht um Faktor 1.2

**Verwendung**: Präzise Zoom-Stufen ohne Maus

### Reset-Button

**🎯 (Reset View)**: Setzt Kamera auf Standardposition zurück

**Funktion**: `ZoomExtents()` - Modell wird automatisch zentriert und in optimaler Größe angezeigt

**Verwendung**:
- Nach zu starkem Zoom "verloren gegangen"
- Nach Rotation Standardansicht wiederherstellen
- Neues Modell optimal anzeigen

---

## Tipps & Tricks

### 1. Navigation optimieren

**Problem**: Modell zu klein/groß

**Lösung**: Verwenden Sie `🎯 Reset View` für optimale Größe

---

### 2. Detail untersuchen

**Workflow**:
1. **Reset View** (🎯) für Überblick
2. **ViewCube** für gewünschte Ansicht klicken
3. **Zoom** mit Mausrad für Detail
4. **Pan** (mittlere Maus) für Positionierung
5. **Rotation** (linke Maus) für feine Anpassung

---

### 3. Anschlusspunkte genau betrachten

**Workflow**:
1. Wählen Sie Anschlusspunkt in der Liste (Sidebar)
2. Zoomen Sie mit Mausrad zum Anschlusspunkt
3. Rotieren Sie mit linker Maus für beste Perspektive
4. Beachten Sie blauen Pfeil für Einsteckrichtung

---

### 4. Vergleich mehrerer Bauteile

**Workflow**:
1. Laden Sie erstes Bauteil
2. Verwenden Sie ViewCube für Standardansicht (z.B. "Front")
3. Merken Sie Position/Zoom
4. Laden Sie zweites Bauteil
5. Verwenden Sie gleiche ViewCube-Ansicht für Vergleich

---

## Performance-Hinweise

### Große Modelle (>50.000 Dreiecke)

Bei sehr großen Modellen kann die Rotation ruckeln:

**Lösungen**:
- Deaktivieren Sie Inertia (falls möglich in Code)
- Reduzieren Sie Rotation Sensitivity
- Schließen Sie andere 3D-intensive Anwendungen

### Viele Anschlusspunkte (>20)

Viele Anschlusspunkte mit Pfeilen und Labels können Performance beeinträchtigen:

**Lösungen**:
- Deaktivieren Sie "Anschlusspunkte anzeigen" während Navigation
- Aktivieren Sie Anschlusspunkte nur zur Inspektion

---

## Tastatur-Shortcuts

**Hinweis**: Aktuell sind keine Tastatur-Shortcuts implementiert.

**Mögliche zukünftige Shortcuts**:
- `R`: Reset View
- `F`: Front View
- `T`: Top View
- `+/-`: Zoom In/Out
- `Leertaste`: Toggle Rotation

---

## Troubleshooting

### Problem: Maus-Rotation funktioniert nicht

**Mögliche Ursachen**:
1. Viewport hat keinen Fokus → Klicken Sie einmal in den Viewport
2. Modell nicht geladen → Wählen Sie Datei aus Liste
3. HelixViewport3D nicht initialisiert → Prüfen Sie Log-Datei

---

### Problem: Mittlere Maustaste funktioniert nicht

**Lösungen**:
1. Prüfen Sie ob Ihre Maus eine funktionsfähige Mitteltaste hat
2. Verwenden Sie alternative Geste (falls implementiert)
3. Verwenden Sie externe Maus mit Mitteltaste

---

### Problem: ViewCube wird nicht angezeigt

**Lösung**:
- ViewCube sollte standardmäßig aktiviert sein (`ShowViewCube="True"`)
- Falls nicht sichtbar, prüfen Sie MainWindow.xaml Zeile 156

---

## Weiterführende Informationen

- HelixToolkit.Wpf Dokumentation: https://github.com/helix-toolkit/helix-toolkit
- HelixViewport3D Beispiele: https://github.com/helix-toolkit/helix-toolkit/wiki

---

## Zusammenfassung

| Aktion | Geste | Beschreibung |
|--------|-------|--------------|
| **Drehen** | Linke Maus + Ziehen | Modell rotieren (Turntable-Modus) |
| **Verschieben** | Mittlere Maus + Ziehen | Kamera horizontal/vertikal bewegen |
| **Zoomen** | Mausrad / Rechte Maus + Ziehen | Vergrößern/Verkleinern |
| **Standardansicht** | ViewCube klicken | Front/Top/Left/etc. Ansicht |
| **Reset** | 🎯 Button | Optimale Ansicht wiederherstellen |

**Die Maussteuerung funktioniert sofort nach dem Start der Anwendung - einfach ausprobieren!**
