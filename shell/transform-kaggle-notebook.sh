#!/bin/bash

# src/hoge/ごとにブロックを作成する
function module_block () {
    n=$(echo $1 | sed -e 's/src\///g' | sed -e 's/\///g' | sed -e 's/.py//g')
    NAME=$(echo ${n^^})
    
    echo "" >> ./kaggle-notebook.py \
    && echo "############################################################" >> ./kaggle-notebook.py \
    && echo "# ${NAME}" >> ./kaggle-notebook.py \
    && echo "############################################################" >> ./kaggle-notebook.py \
    && echo "" >> ./kaggle-notebook.py
}

# 初期化
echo > ./kaggle-notebook.py

# src/hoge/以下のファイルを展開
DIRS=$(echo $(ls -d src/*/))
for d in ${DIRS}
do
    if [ ! "`echo ${d} | grep __pycache__`" ]; then 
        module_block ${d}
        FILES=$(echo $(ls -d ${d}*))
        for f in ${FILES}
        do
            if [ ! "`echo ${f} | grep __pycache__`" ]; then 
                cat ${f} >> ./kaggle-notebook.py
            fi
        done
    fi
done

# src/直下のファイルを展開
FILES=$(echo $(ls -d src/*.py))
for f in ${FILES}
do
    if [ ! "`echo ${f} | grep __init__.py`" ]; then
        module_block ${f}
    fi
    cat ${f} >> ./kaggle-notebook.py
done

# mainファイルを展開
module_block RUN
cat run.py >> ./kaggle-notebook.py
