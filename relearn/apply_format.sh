#!/bin/bash

mkdir format/
cd source

for filename in ./*
do
clang-format-10 -style=file $filename > ../format/$filename
done

cd ../format
for filename in ./*
do
cp $filename ../source/$filename
rm $filename
done

cd ..
rmdir format
