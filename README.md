# CineBo

Eine Kamera-App für das OPPO Find N6, die beim Filmen live ein natürliches Bokeh erzeugt. Die Schärfe liegt automatisch oder per Tippen auf Personen und Objekten.

## Stand

Phase 1: Machbarkeitstest.

- Stufe 1 (bestanden): Kamera, 4K-Aufnahme, Klappen und Wärme.
- Stufe 2 (läuft): Schaffen beide KI-Modelle (Tiefe und Verfolgung) auf dem KI-Chip das nötige Tempo, auch während 4K-Aufnahme und über 10 Minuten?

## Test-App

GitHub baut jede Änderung automatisch. Die neueste Version liegt immer unter
[cinebo-test.apk](https://github.com/botrik-de/cinebo/releases/latest/download/cinebo-test.apk).

## KI-Modelle

Die Modelle liegen nie im Repo, sie kommen beim Bauen dazu:

- Depth Anything V2 Small (Tiefe) und EdgeTAM (Verfolgung) in der Fassung von [Qualcomm AI Hub](https://github.com/quic/ai-hub-models).
- Die Teile von EdgeTAM, die Qualcomm nicht anbietet (Memory-Attention und Memory-Encoder), baut `tools/export/edgetam_export.py` aus dem [Original von Meta](https://github.com/facebookresearch/EdgeTAM) nach, prüft sie gegen das Original und legt sie im Release `modelle-1` ab.

Lizenzen: EdgeTAM und Depth Anything V2 Small stehen unter Apache 2.0. Qualcomms QNN-Bibliotheken dürfen laut ihrer Lizenz eingebaut in eine App weitergegeben werden.

## Hinweise

- Das Projekt ist öffentlich, damit die Builds kostenlos bleiben. Testvideos und Aufnahmen liegen hier nie.
- Der Schlüssel im Ordner `signing` gilt nur für die Test-App und ist absichtlich öffentlich. Die richtige App bekommt einen eigenen, geheimen Schlüssel.
- Für den Code dieses Projekts gibt es keine Lizenz. Alle Rechte bleiben beim Inhaber dieses Kontos.
