#!/bin/bash
competition=""  # Fill comepetiton name

if [ "$1" == "" ]; then
  ls_submit=$(ls -r submit)
  submits=(${ls_submit// / })
  arg=$(echo ${submits[0]} | sed 's/submission_//g' | sed 's/.csv//g')
else
  arg=$1
fi
ymdhms=(${arg//-/ })
ymd=${ymdhms[0]}-${ymdhms[1]}-${ymdhms[2]}
hms=${ymdhms[3]}-${ymdhms[4]}-${ymdhms[5]}
filepath=./submit/submission_${ymd}-${hms}.csv
message=$(cat ./outputs/${ymd}/${hms}/main.log | grep RMSE | sed 's/ //g' | sed 's/\t//g')

echo filepath: ${filepath}
echo message: ${message}

kaggle competitions submit -c ${competition} -f ${filepath} -m ${message}
