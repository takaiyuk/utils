#!/bin/bash

sudo docker run -d --name $1 -p 8888:8888 -v ${PWD}:/workspace -v ${HOME}/.kaggle:/root/.kaggle/ takaiyuk/$2:latest
