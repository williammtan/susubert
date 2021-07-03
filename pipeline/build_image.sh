
comp_dir=$1
image_base="gcr.io/food-id-app"
image_namespace="susubert/$comp_dir" # eg. susubert/
image_tag="latest"
full_image="$image_base/$image_namespace"

docker build -t $image_namespace $comp_dir
docker tag $image_namespace:$image_tag $full_image
docker push $full_image

docker inspect --format="{{index .RepoDigests 0}}" $full_image
