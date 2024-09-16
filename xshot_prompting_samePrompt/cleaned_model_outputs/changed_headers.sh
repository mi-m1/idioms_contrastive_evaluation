var="idiom,pred"
for i in *.csv;
do sed -i "1s/.*/$var/" $i;
done