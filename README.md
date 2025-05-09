Ich werde die README auf Deutsch übersetzen, während ich die englischen Parameter, Funktionsnamen und Codebeispiele unverändert lasse.

# B. subtilis Kompetenz-Schaltkreis Analyse
Derya Kilicarslan

Dieses Repository enthält Python-Skripte zur Analyse der Kompetenz-Schaltkreisdynamik in B. subtilis Bakterien.

## Projektstruktur

Das Projekt ist in folgende Module gegliedert:

- `competence_circuit_analysis.py` - Kern-Modelldefinitionen und Analysefunktionen
- `helpers.py` - Allgemeine Hilfsfunktionen
- `visualization.py` - Visualisierungsfunktionen zum Erstellen von Diagrammen
- `simulation.py` - Funktionen für deterministische und stochastische Simulationen
- `parameter_search.py` - Werkzeuge zur Suche nach erregbaren Konfigurationen im Parameterraum
- `run_analysis.py` - Hauptskript, das alles zusammenführt und Anpassungen ermöglicht

## Erste Schritte

### Voraussetzungen

Der Code benötigt folgende Python-Pakete:

```
numpy
matplotlib
pandas
scipy
scikit-learn
```

Sie können diese mit pip installieren:

```bash
pip install numpy matplotlib pandas scipy scikit-learn
```

### Ausführen der Analyse

Sie können die Analyse auf verschiedene Weise ausführen:

#### 1. Standardanalyse

Um die vollständige Analyse mit Standardeinstellungen auszuführen, führen Sie einfach aus:

```bash
python run_analysis.py
```

#### 2. Verwendung einer Konfigurationsdatei

Sie können die Analyse anpassen, indem Sie eine JSON-Konfigurationsdatei erstellen:

```bash
python run_analysis.py --config analysis_config.json
```

Siehe die Beispieldatei `analysis_config.json` für verfügbare Parameter.

#### 3. Kommandozeilen-Optionen

Sie können bestimmte Analysen ausführen und Parameter direkt über die Kommandozeile festlegen:

```bash
# Nur Standardparameteranalyse und stochastische Analyse ausführen
python run_analysis.py --standard --stochastic

# Suel-Parametersuche mit höherer Auflösung ausführen
python run_analysis.py --suel --suel-points 150

# Alle Analysen mit angepassten stochastischen Simulationsparametern ausführen
python run_analysis.py --all --stochastic-sims 50 --stochastic-time 500
```

#### Verfügbare Kommandozeilen-Optionen

**Modulauswahl:**
- `--all`: Alle Analysen ausführen
- `--standard`: Standardparameteranalyse ausführen
- `--excitable`: Suche nach erregbaren Konfigurationen ausführen
- `--hill`: Hill-Koeffizienten-Analyse ausführen
- `--suel`: Suel-Parameterraumsuche ausführen
- `--stochastic`: Stochastische Analyse ausführen
- `--amplification`: Verstärkungsfaktoranalyse ausführen

**Parameter-Überschreibungen:**
- `--output-prefix`: Präfix für Ausgabeverzeichnis
- `--excitable-samples`: Anzahl der Stichproben für die Suche nach Erregbarkeit
- `--hill-grid-size`: Rastergröße für die Hill-Koeffizienten-Suche
- `--suel-points`: Anzahl der Punkte für die Suel-Suche
- `--stochastic-sims`: Anzahl der stochastischen Simulationen
- `--stochastic-time`: Maximale Zeit für stochastische Simulationen
- `--amp-sims`: Anzahl der Verstärkungssimulationen
- `--amp-time`: Maximale Zeit für Verstärkungssimulationen

## Konfigurationsoptionen

Sie können die Analyse anpassen, indem Sie folgende Parameter ändern:

### Allgemeine Parameter
- `output_prefix`: Präfix für den Namen des Ausgabeverzeichnisses

### Aktivieren/Deaktivieren von Modulen
- `run_standard_analysis`: Ob Standardparameter analysiert werden sollen
- `run_excitable_search`: Ob nach erregbaren Konfigurationen gesucht werden soll
- `run_hill_analysis`: Ob Hill-Koeffizienten-Effekte analysiert werden sollen
- `run_suel_search`: Ob der Suel-Parameterraum durchsucht werden soll
- `run_stochastic_analysis`: Ob stochastische Simulationen durchgeführt werden sollen
- `run_amplification_analysis`: Ob Verstärkungsfaktoren analysiert werden sollen

### Analysespezifische Parameter
- `standard_t_max`: Maximale Zeit für Simulationen mit Standardparametern
- `excitable_n_samples`: Anzahl der Stichproben für die Suche nach Erregbarkeit
- `hill_grid_size`: Rastergröße für die Hill-Koeffizienten-Suche
- `hill_extended_search`: Ob eine erweiterte Suche für Hill-Koeffizienten durchgeführt werden soll
- `suel_n_points`: Anzahl der Punkte pro Dimension für die Suel-Suche
- `stochastic_n_simulations`: Anzahl der stochastischen Simulationen pro Parametersatz
- `stochastic_t_max`: Maximale Zeit für stochastische Simulationen
- `stochastic_amplification`: Rauschverstärkungsfaktor für stochastische Simulationen
- `amplification_factors`: Liste der zu analysierenden Verstärkungsfaktoren
- `amplification_n_simulations`: Anzahl der Simulationen pro Verstärkungsfaktor
- `amplification_t_max`: Maximale Zeit für Verstärkungsfaktor-Simulationen

## Beispiel-Konfigurationsdatei

Hier ist ein Beispiel für eine Konfigurationsdatei:

```json
{
    "output_prefix": "competence_analysis_custom",
    
    "run_standard_analysis": true,
    "run_excitable_search": true,
    "run_hill_analysis": true,
    "run_suel_search": true,
    "run_stochastic_analysis": true,
    "run_amplification_analysis": true,
    
    "standard_t_max": 150,
    
    "excitable_n_samples": 25000,
    
    "hill_grid_size": 30,
    "hill_extended_search": true,
    
    "suel_n_points": 75,
    
    "stochastic_n_simulations": 30,
    "stochastic_t_max": 300,
    "stochastic_amplification": 10,
    
    "amplification_factors": [1, 2, 5, 10, 15],
    "amplification_n_simulations": 25,
    "amplification_t_max": 300
}
```
