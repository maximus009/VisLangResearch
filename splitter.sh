# Use FFMPEG to read each video and split into frame sequences.
# Will sample 2 frames every second

# -- 10/12/17 9am

# tested on Macsimus. Works.
# using ffmpeg version 3.1.3

mkdir -p frames

for videoName in "$1"/*
do
    baseName="$(basename $videoName)"
    baseName="${baseName%.*}"
    mkdir -p frames/$baseName
    ffmpeg -i $videoName -r 2 frames/$baseName/frame_%3d.png
done
