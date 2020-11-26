echo "Starting...."
rm -rf output
mkdir output/violin/ output/guitar/ output/flute/
python analyze.py -i input/violin.wav -o output/violin/
python analyze.py -i input/flute.mp3 -o output/flute/
python analyze.py -i input/guitar.wav -o output/ssguitar/
echo "Eng of prog"