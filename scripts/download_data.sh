#!/usr/bin/env bash
# Scarica i dati BreastDCEDL da Zenodo (record 18114231).
#
# Uso (dalla cartella principale della repo):
#   bash scripts/download_data.sh          -> metadati CSV + dati demo (~55 MB)
#   bash scripts/download_data.sh ispy1    -> coorte I-SPY1 MinCrop (~1,3 GB)
#   bash scripts/download_data.sh ispy2    -> coorte I-SPY2 MinCrop (~13,6 GB)
#
# Gli archivi vengono estratti mentre si scaricano (streaming): il file .tar.gz
# non viene mai salvato su disco, quindi serve solo lo spazio dei file estratti.
# Se la connessione cade, cancella la cartella estratta a meta' e rilancia.

set -euo pipefail

BASE="https://zenodo.org/records/18114231/files"
DATA_DIR="data"
mkdir -p "$DATA_DIR"

download_stream () {
    local file="$1"
    local dest="$2"
    mkdir -p "$dest"
    echo ">> Scarico ed estraggo $file in $dest ..."
    curl -fL --retry 3 --progress-bar "$BASE/$file?download=1" | tar -xzf - -C "$dest"
    echo ">> Fatto: $file"
}

target="${1:-base}"

case "$target" in
    base)
        echo ">> Scarico i metadati CSV ..."
        curl -fL --retry 3 --progress-bar \
            "$BASE/BreastDCEDL_metadata_min_crop.csv?download=1" \
            -o "$DATA_DIR/BreastDCEDL_metadata_min_crop.csv"
        download_stream "BreastDCEDL_demo_data.tar.gz" "$DATA_DIR/demo"
        ;;
    ispy1)
        download_stream "BreastDCEDL_ISPY1_min_crop.tar.gz" "$DATA_DIR/ispy1"
        ;;
    ispy2)
        download_stream "BreastDCEDL_ISPY2_min_crop.tar.gz" "$DATA_DIR/ispy2"
        ;;
    *)
        echo "Argomento non valido: $target (usa: base | ispy1 | ispy2)"
        exit 1
        ;;
esac

echo
echo ">> Spazio occupato in $DATA_DIR:"
du -sh "$DATA_DIR"/* 2>/dev/null || true
echo ">> Spazio libero sul disco:"
df -h . | tail -1
