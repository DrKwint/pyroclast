dir=$1

for f in $(ls ${dir}/*.dot); do
    echo $f
    dot -Tpng ${f} -o ${f%.dot}.png
done