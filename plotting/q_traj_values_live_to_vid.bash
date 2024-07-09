DIR=$1

ffmpeg -framerate 15 -pattern_type glob -i "${DIR}/*.png" -vcodec libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -r 30 "${DIR}-plot.mp4"