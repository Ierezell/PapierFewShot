#!/bin/zsh
while :
do
	ram="$(ps aux | grep python | awk '{sum+=$6} END {print int(sum/1024)}')"
	echo "$ram / $1 Mb"
	if [ $ram -gt $1 ]
	then
		echo "Tué"
		pkill python
	fi
	sleep 5
done
