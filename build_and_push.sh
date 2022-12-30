#!/bin/bash

AWS_REGION=ap-northeast-2
AWS_ECR_URI=008747557116.dkr.ecr.ap-northeast-2.amazonaws.com
IMAGE_NAME=naily-style-classify-server
TAG_NAME=latest

DOCKER_IMAGE=$IMAGE_NAME:$TAG_NAME
AWS_ECR_FULL_URI=$AWS_ECR_URI/$DOCKER_IMAGE

echo "remove image"
docker rmi -f $AWS_ECR_FULL_URI
docker rmi -f $DOCKER_IMAGE

echo "build image"
docker build -t $DOCKER_IMAGE .

echo "login aws ecr"
aws ecr get-login-password --region $AWS_REGION | \
docker login --username AWS --password-stdin $AWS_ECR_URI

echo "docker push"
docker image tag $DOCKER_IMAGE $AWS_ECR_FULL_URI
docker push $AWS_ECR_FULL_URI