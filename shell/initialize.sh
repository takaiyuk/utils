#!/bin/bash

function makedirs () {
    mkdir -p data/external
    mkdir -p data/interim
    mkdir -p data/processed
    mkdir -p data/raw

    mkdir -p docker

    mkdir -p docs

    mkdir -p models/adversarial
    mkdir -p models/efficientdet
    mkdir -p models/importance
    mkdir -p models/model

    mkdir -p notebooks

    mkdir -p references

    mkdir -p shell

    mkdir -p src/data
    mkdir -p src/features
    mkdir -p src/models
    mkdir -p src/visualization
    mkdir -p src/submit
    mkdir -p src/utils

    mkdir -p submissions
}

function touch_keep () {
    touch data/external/.gitkeep
    touch data/interim/.gitkeep
    touch data/processed/.gitkeep
    touch data/raw/.gitkeep

    touch docker/.gitkeep

    touch docs/.gitkeep

    touch models/adversarial/.gitkeep
    touch models/efficientdet/.gitkeep
    touch models/importance/.gitkeep
    touch models/model/.gitkeep

    touch notebooks/.gitkeep

    touch references/.gitkeep

    touch shell/.gitkeep

    touch src/data/.gitkeep
    touch src/features/.gitkeep
    touch src/models/.gitkeep
    touch src/visualization/.gitkeep
    touch src/submit/.gitkeep
    touch src/utils/.gitkeep

    touch submissions/.gitkeep
}

function touch_init () {
    touch src/__init__.py
    touch src/data/__init__.py
    touch src/features/__init__.py
    touch src/models/__init__.py
    touch src/visualization/__init__.py
    touch src/submit/__init__.py
    touch src/utils/__init__.py

    touch docker/pull.sh
    touch docker/run.sh
    touch docker/exec.sh
    touch docs/competition.md
    touch shell/download.sh
    touch shell/submit.sh
    touch src/const.py
    touch .gitignore
    touch config.yml
    touch README.md
    touch run.sh
}

function git_init () {
    git init
}

makedirs
touch_keep
touch_init
git_init

