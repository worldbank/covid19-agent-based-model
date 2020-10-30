conda env export -n covid19_abm --no-builds | grep -v "prefix" > environment.yml
