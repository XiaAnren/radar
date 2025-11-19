#!/bin/bash

start="2024-10-17 06:00:00" end="2024-10-17 18:00:00"
savepath="data/GPM"

current=$(date -d "$start" +%s)
[[ ! -d $savepath ]] && mkdir $savepath;  # cd $savepath
while [[ $current -le $(date -d "$end" +%s) ]]; do
  suffix=$(printf "%04d" $(date -d "@$current" +"%H * 60 + %M" | bc))
  filepath="https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/$(date -d "@$current" +"%Y")"
  folder=$(date -d "$(date -d "@$current" +"%Y-%m-%d")" +%j)
  filename="3B-HHR.MS.MRG.3IMERG.$(date -d "@$current" +"%Y%m%d-S%H%M%S")-$(date -d "@$((current + 1800 - 1))" +"E%H%M%S").$suffix.V07B.HDF5"

  if [[ ! -e $savepath/$filename ]]; then
    command="wget --quiet -P $savepath $filepath/$folder/$filename"
    $command && echo "Success: $command" || { echo "Error: $command"; exit 1; }
  fi

  current=$((current + 1800))
done
