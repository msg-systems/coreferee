#!/bin/bash

# This script downloads the training corpora for English, German and Polish. Instructions for French are at the bottom. 
#
# Usage: download_corpora.sh <target dir>
set -e
if [ $# -ne 1 ]
then
  echo "Usage: download_corpora.sh <target dir>"
  exit 1
fi
DATA_DIR=$1

mkdir ${DATA_DIR} # exit if data directory already exists

# Set up temp directory
TEMP_DIR="${DATA_DIR}/tmp"
mkdir ${TEMP_DIR}
pushd ${TEMP_DIR}

# Download English and German files from ParCor corpus

mkdir -p "${DATA_DIR}/en"
mkdir -p "${DATA_DIR}/de"
wget -O ParCor_v1.0.tar.gz https://opus.nlpl.eu/download.php?f=ParCor/ParCor_v1.0.tar.gz 
tar -xvzf ParCor_v1.0.tar.gz
find "${TEMP_DIR}/ParCor" -type f | grep "English/Annotator1" | grep ".xml" | xargs -I{} mv {} "${DATA_DIR}/en"
find "${TEMP_DIR}/ParCor" -type f | grep "German/Annotator1" | grep ".xml" | xargs -I{} mv {} "${DATA_DIR}/de"

# Download English LitBank corpus

git clone https://github.com/dbamman/litbank
cp litbank/coref/brat/* "${DATA_DIR}/en"

# Download Polish Coreference Corpus

git clone https://github.com/dbamman/litbank
cp litbank/coref/brat/* "${DATA_DIR}/en"

# Clean up
popd
rm -Rf ${TEMP_DIR}

# Instructions for French
# - download DEMOCRAT corpus from https://www.ortolang.fr/market/corpora/democrat/
# - convert it to CONLL using https://github.com/Pantalaymon/neuralcoref-for-french/blob/main/conversion_conll.py