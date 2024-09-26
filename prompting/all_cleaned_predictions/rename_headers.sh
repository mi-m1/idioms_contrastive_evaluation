var="idiom,pred"
# sed -i "1s/.*/$var/" file.txt


for i in *.csv; do sed -i "1s/.*/$var/" $i; done