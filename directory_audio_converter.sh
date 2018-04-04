# Batch ffmpeg converter of sph to mp3
# Dependencies: ffmpeg

# to use:
# Step 1: cd into directory of .sph folders
# Step 2: 'bash ../../sph_to_mp3_converter.sh'

# mkdir -p -- "./output"; # makes folder if it doesnt exist

# converts all .sph files into mp3
for i in $(find */* -name '*.WAV' -maxdepth 4 -type f); do
# for i in *.flac;
# find  /* -type f -maxdepth 4 | while read file; do
  ffmpeg -y -i "$i" -ar 16000 "${i%.WAV}.wav";
done

# moves all .mp3 files into new folder
# mkdir -p -- "./output"; # makes folder if it doesnt exist
# # mv **/*.mp3 ./output;
# find . -name '*.mp3' -exec mv {} ./output \;
