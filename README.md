# zinc
Zeldovich Initial Condition for N-body simulations. 

To use zinc, you require an installation of [```classylss```](https://github.com/nickhand/classylss) and this clone of [```LensTools```](https://github.com/utkarshgiri/LensTools)

# Docker

Alternatively, you can run zinc in Docker buy first building it using
```bash
export DOCKER_BUILDKIT=1
docker build --ssh github=/PATH/TO/.ssh/id_rsa -t zinc .
```
You can run it with
```bash
docker run -it --rm zinc bash
```

