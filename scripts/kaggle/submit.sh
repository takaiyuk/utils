#!/bin/bash
COMPETITION_NAME=""  # Fill comepetiton name

EXP=$1
# if [ -n "$EXP" ]; then
#   echo "EXP should be provided"
#   exit 1
# fi
echo ${EXP}
FILEPATH=output/submission/submission_${EXP}.csv
MESSAGE=$(cat ./output/logs/${EXP}/result.log | sed 's/ //g' | sed 's/\t//g')
echo filepath: ${FILEPATH}
echo message: ${MESSAGE}

kaggle competitions submit -c ${COMPETITION_NAME} -f ${FILEPATH} -m "${MESSAGE}"
