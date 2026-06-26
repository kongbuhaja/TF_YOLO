sudo docker build -f "${PWD}/env.dockerfile" -t tf_yolo:2.0 .
sudo docker run --gpus '"device=0"' -m 64g --shm-size=8g -it -v "${PWD%/*/*}":/home/ML -e HOST_PERMS="$(id -u):$(id -g)" --name tf_yolo2 tf_yolo:2.0
