#!/usr/bin/env bash
# Download the crowd-enVent corpus (Troiano, Oberlaender & Klinger 2023, Computational Linguistics 49(1)).
# No licence is stated for the corpus: it is git-ignored, never commit it.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/crowd-enVent
curl -L -o data/crowd-enVent/crowd-enVent2023.zip https://www.romanklinger.de/data-sets/crowd-enVent2023.zip
unzip -o data/crowd-enVent/crowd-enVent2023.zip -d data/crowd-enVent/
echo "Corpus in data/crowd-enVent/corpus/"
