for f in $(ls *.dot); do
    dot -Tpng $f -o ${f%.dot}.png
done