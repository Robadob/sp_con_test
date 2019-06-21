#!/bin/bash

#NVCC
nvcc main.cu -I ../include/ -I . -std=c++11 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -rdc=false -m 64 -g -G -D _DEBUG -o debug-sp_con_test
