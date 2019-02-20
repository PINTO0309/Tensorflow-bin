#!/bin/sh
### BEGIN INIT INFO
# Provides:       zram
# Required-Start:
# Required-Stop:
# Default-Start:  2 3 4 5
# Default-Stop:   0 1 6
### END INIT INFO

case "$1" in
	start)
		modprobe zram

		echo lz4 > /sys/block/zram0/comp_algorithm
		echo 512M > /sys/block/zram0/disksize

		mkswap /dev/zram0
		swapon -p 5 /dev/zram0
		;;
	stop)
		swapoff /dev/zram0
		sleep 1
		modprobe -r zram
		;;
	*)
		echo "Usage $0 start | stop "
		;;
esac
