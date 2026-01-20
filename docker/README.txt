#Build dell'immagine Docker:

docker build -t sintesi-app .

#Esecuzione del Container:

docker run --gpus all -p 8080:8080 sintesi-app
