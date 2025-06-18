sudo docker build -f "${PWD}/env.dockerfile" -t tf_yolo:1.0 .
# sudo docker run --gpus '"device=0"' --cpuset-cpus=16-23 -m 16g --shm-size=8g -it -v "${PWD%/*/*}":/home/ML --name test tf_yolo:1.0
sudo docker run --cpuset-cpus=16-23 -m 16g --shm-size=8g -it -v "${PWD%/*/*}":/home/ML -e HOST_PERMS="$(id -u):$(id -g)" --name test tf_yolo:1.0
