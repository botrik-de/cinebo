#!/usr/bin/env python3
"""KI-Dateien der Test-App: Liste bauen, gegen Release und Archive prüfen, zusammenfassen.

    ki_dateien.py packen     ROH LADEN BASIS SATZ [--archiv DATEI URL PRAEFIX NAME...]...
    ki_dateien.py pruefen    LISTE ROH RELEASE BASIS [ARCHIV-DATEI...]
    ki_dateien.py uebersicht LISTE
    ki_dateien.py apk        APK LISTE MAX-MB ORDNER...

packen: Jede Datei in ROH wird eine Quelle in unserem Release (Adresse BASIS + Name). Gepackt
(gzip) wird sie, wenn das mindestens 5 % spart; die Übertragung landet in LADEN. Ein Archiv
(etwa Qualcomms QNN-Paket) bleibt, wo es ist: Die App lädt es von URL und entpackt daraus die
Dateien PRAEFIX/NAME. Die Liste kommt auf die Standardausgabe.

pruefen: Prüft die Liste gegen die fertigen Dateien in ROH, die Übertragungen im Ordner
RELEASE (so, wie sie wirklich im Release liegen) und die Archive, alles so, wie es die App tut.

apk: Wacht darüber, dass die APK klein bleibt, nichts doppelt enthält, was die App selbst
lädt, und dass jede Bibliothek auf dem Handy findet, was sie braucht.
"""
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

MINDEST_ERSPARNIS = 0.05

# Was Android jeder App mitgibt; libcdsprpc.so gibt der Hersteller frei (uses-native-library).
SYSTEM_BIBLIOTHEKEN = {
    'libc.so', 'libm.so', 'libdl.so', 'liblog.so', 'libz.so', 'libandroid.so', 'libjnigraphics.so',
    'libEGL.so', 'libGLESv2.so', 'libGLESv3.so', 'libvulkan.so', 'libnativewindow.so', 'libmediandk.so',
    'libcamera2ndk.so', 'libOpenSLES.so', 'libaaudio.so', 'libsync.so', 'libbinder_ndk.so',
    'libneuralnetworks.so', 'libstdc++.so', 'libcdsprpc.so',
}


def sha256(daten):
    return hashlib.sha256(daten).hexdigest()


def lesen(pfad):
    with open(pfad, 'rb') as f:
        return f.read()


def liste_lesen(pfad):
    with open(pfad, encoding='utf-8') as f:
        return json.load(f)


def fehler_melden(fehler):
    if fehler:
        for f in fehler:
            print('FEHLER: ' + f, file=sys.stderr)
        sys.exit(1)


def packen(argumente):
    roh, laden, basis, satz = argumente[:4]
    archive = []
    rest = argumente[4:]
    while rest:
        if rest[0] != '--archiv' or len(rest) < 5:
            raise SystemExit('--archiv braucht DATEI URL PRAEFIX NAME...')
        ende = next((i for i in range(1, len(rest)) if rest[i] == '--archiv'), len(rest))
        archive.append((rest[1], rest[2], rest[3], rest[4:ende]))
        rest = rest[ende:]
    os.makedirs(laden, exist_ok=True)
    quellen, dateien = [], []
    for name in sorted(os.listdir(roh)):
        daten = lesen(os.path.join(roh, name))
        gepackt = gzip.compress(daten, compresslevel=9, mtime=0)
        gz = len(daten) > 0 and len(gepackt) <= len(daten) * (1 - MINDEST_ERSPARNIS)
        lade_name, lade_daten = (name + '.gz', gepackt) if gz else (name, daten)
        with open(os.path.join(laden, lade_name), 'wb') as f:
            f.write(lade_daten)
        quellen.append({'name': lade_name, 'url': basis + lade_name, 'bytes': len(lade_daten),
                        'sha256': sha256(lade_daten), 'gzip': gz})
        dateien.append({'name': name, 'bytes': len(daten), 'sha256': sha256(daten), 'quelle': lade_name})
    for datei, url, praefix, namen in archive:
        archiv = lesen(datei)
        name = os.path.basename(datei)
        quellen.append({'name': name, 'url': url, 'bytes': len(archiv), 'sha256': sha256(archiv), 'archiv': True})
        with zipfile.ZipFile(datei) as z:
            for n in namen:
                pfad = praefix.rstrip('/') + '/' + n
                inhalt = z.read(pfad)
                dateien.append({'name': n, 'bytes': len(inhalt), 'sha256': sha256(inhalt), 'quelle': name, 'pfad': pfad})
    namen = [d['name'] for d in dateien]
    if len(set(namen)) != len(namen):
        raise SystemExit('Doppelte Dateinamen in der Liste: %s' % sorted({n for n in namen if namen.count(n) > 1}))
    json.dump({'version': satz, 'quellen': quellen, 'dateien': dateien}, sys.stdout, indent=1, ensure_ascii=False)
    print()


def pruefen(liste_pfad, roh, release, basis, *archiv_dateien):
    liste = liste_lesen(liste_pfad)
    quellen = {q['name']: q for q in liste.get('quellen', [])}
    dateien = liste.get('dateien', [])
    archive = {os.path.basename(a): a for a in archiv_dateien}
    fehler = []

    # Jede Datei aus unserem Release liegt fertig in ROH, und umgekehrt.
    eigene = sorted(d['name'] for d in dateien if not quellen.get(d['quelle'], {}).get('archiv'))
    if eigene != sorted(os.listdir(roh)):
        fehler.append('Liste und fertige Dateien passen nicht zusammen: nur in der Liste %s, nur als Datei %s'
                      % (sorted(set(eigene) - set(os.listdir(roh))), sorted(set(os.listdir(roh)) - set(eigene))))
    genutzt = {d['quelle'] for d in dateien}
    if genutzt != set(quellen):
        fehler.append('Quellen ohne Datei oder Dateien ohne Quelle: %s' % sorted(genutzt ^ set(quellen)))

    erwartet = {'liste.json'}
    for q in quellen.values():
        if q.get('archiv'):
            pfad = archive.get(q['name'])
            if pfad is None:
                fehler.append('%s: Archiv zum Prüfen fehlt' % q['name'])
                continue
        else:
            if q['url'] != basis + q['name']:
                fehler.append('%s: Adresse %s zeigt nicht ins Release %s' % (q['name'], q['url'], basis))
                continue
            erwartet.add(q['name'])
            pfad = os.path.join(release, q['name'])
            if not os.path.exists(pfad):
                fehler.append('%s fehlt im Release' % q['name'])
                continue
        daten = lesen(pfad)
        if len(daten) != q['bytes'] or sha256(daten) != q['sha256']:
            fehler.append('%s: anders als in der Liste' % q['name'])
            continue
        for d in (d for d in dateien if d['quelle'] == q['name']):
            if q.get('archiv'):
                with zipfile.ZipFile(pfad) as z:
                    fertig = z.read(d['pfad'])
            elif q.get('gzip'):
                fertig = gzip.decompress(daten)
            else:
                fertig = daten
            if len(fertig) != d['bytes'] or sha256(fertig) != d['sha256']:
                fehler.append('%s ergibt nicht die fertige Datei %s' % (q['name'], d['name']))
            if not q.get('archiv'):
                roh_pfad = os.path.join(roh, d['name'])
                if os.path.exists(roh_pfad) and sha256(lesen(roh_pfad)) != d['sha256']:
                    fehler.append('%s: fertige Datei weicht von der Liste ab' % d['name'])
    uebrig = sorted(set(os.listdir(release)) - erwartet)
    if uebrig:
        print('Hinweis: Im Release liegen Dateien, die nicht in der Liste stehen: %s' % uebrig)
    fehler_melden(fehler)
    print('KI-Satz %s: %d Quellen und %d Dateien geprüft, alles stimmt mit der Liste überein.'
          % (liste.get('version'), len(quellen), len(dateien)))


def uebersicht(liste_pfad):
    liste = liste_lesen(liste_pfad)
    fertig = sum(d['bytes'] for d in liste['dateien'])
    lade = sum(q['bytes'] for q in liste['quellen'])
    print('%-40s %14s  %s' % ('Quelle', 'Übertragung', 'woher'))
    for q in liste['quellen']:
        woher = re.sub(r'^https?://([^/]+)/.*$', r'\1', q['url'])
        art = 'Archiv' if q.get('archiv') else ('gzip' if q.get('gzip') else 'roh')
        print('%-40s %11.1f MB  %s, %s' % (q['name'], q['bytes'] / 1e6, woher, art))
    print()
    print('%-40s %14s  %s' % ('Datei auf dem Handy', 'Größe', 'aus'))
    for d in liste['dateien']:
        print('%-40s %11.1f MB  %s' % (d['name'], d['bytes'] / 1e6, d['quelle']))
    print()
    print('Zusammen: %.1f MB Übertragung, %.1f MB auf dem Handy' % (lade / 1e6, fertig / 1e6))


def benoetigt(pfad):
    ausgabe = subprocess.run(['readelf', '-dW', pfad], capture_output=True, text=True, check=True).stdout
    return re.findall(r'\(NEEDED\)\s+Shared library: \[([^\]]+)\]', ausgabe)


def apk(apk_pfad, liste_pfad, max_mb, *ordner):
    fehler = []
    groesse = os.path.getsize(apk_pfad)
    if groesse > float(max_mb) * 1e6:
        fehler.append('Die APK ist %.1f MB groß, erlaubt sind höchstens %s MB' % (groesse / 1e6, max_mb))
    nachgeladen = {}
    for o in ordner:
        for n in os.listdir(o):
            if n.endswith('.so'):
                nachgeladen[n] = os.path.join(o, n)
    liste = liste_lesen(liste_pfad)
    fehlend = sorted(set(nachgeladen) - {d['name'] for d in liste['dateien']})
    if fehlend:
        fehler.append('Diese Bibliotheken fehlen in der Liste: %s' % fehlend)
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(apk_pfad) as z:
        namen = z.namelist()
        libs = {n.rsplit('/', 1)[-1]: n for n in namen if n.startswith('lib/arm64-v8a/') and n.endswith('.so')}
        doppelt = sorted(set(libs) & set(nachgeladen))
        if doppelt:
            fehler.append('Diese Bibliotheken lädt die App selbst, sie gehören nicht in die APK: %s' % doppelt)
        if 'assets/modelle.json' not in namen:
            fehler.append('assets/modelle.json fehlt in der APK')
        elif json.loads(z.read('assets/modelle.json')) != liste:
            fehler.append('assets/modelle.json in der APK weicht von der Liste ab')
        pruefen_auf = []
        for name, eintrag in libs.items():
            z.extract(eintrag, tmp)
            pruefen_auf.append((name, os.path.join(tmp, eintrag)))
        pruefen_auf += sorted(nachgeladen.items())
        # Skel-Dateien laufen auf dem KI-Chip selbst; ihre Abhängigkeiten löst der Chip auf.
        pruefen_auf = [(n, p) for n, p in pruefen_auf if 'Skel' not in n]
        vorhanden = set(libs) | set(nachgeladen) | SYSTEM_BIBLIOTHEKEN
        for name, pfad in pruefen_auf:
            for b in benoetigt(pfad):
                if b not in vorhanden:
                    fehler.append('%s braucht %s, das es auf dem Handy nicht gibt' % (name, b))
    fehler_melden(fehler)
    print('APK: %.1f MB, %d eigene Bibliotheken, %d lädt die App selbst; alle Abhängigkeiten erfüllt.'
          % (groesse / 1e6, len(libs), len(nachgeladen)))


if __name__ == '__main__':
    befehl = sys.argv[1] if len(sys.argv) > 1 else ''
    argumente = sys.argv[2:]
    if befehl == 'packen' and len(argumente) >= 4:
        packen(argumente)
    elif befehl == 'pruefen' and len(argumente) >= 4:
        pruefen(*argumente)
    elif befehl == 'uebersicht' and len(argumente) == 1:
        uebersicht(*argumente)
    elif befehl == 'apk' and len(argumente) >= 4:
        apk(*argumente)
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
