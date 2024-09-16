#!/bin/bash

var="idiom,sentence"
# sed -i "1s/.*/$var/" file.txt

for f in *.csv;
do
    sed -i "1s/.*/$var/" $f;
done

