#!/bin/bash

COMPETITION_NAME=$1
if [ -n "$COMPETITION_NAME" ]; then
  echo ${COMPETITION_NAME}
  kaggle competitions download ${COMPETITION_NAME}} -p .
  unzip ${COMPETITION_NAME}}.zip -d data/raw
  rm ${COMPETITION_NAME}}.zip
else
  echo "COMPETITION_NAME should be provided"
  exit 1
fi
