#!/bin/bash
input="running"
while IFS= read -r line
do
  pkill -f "$line"
done < "$input"

killall serpent

echo "5" > /home/dan/.wine/drive_c/input.txt