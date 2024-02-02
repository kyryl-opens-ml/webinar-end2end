docker build -t kyrylprojector/data:latest --target data .
docker push kyrylprojector/data:latest

docker build -t kyrylprojector/experiments:latest --target experiments .
docker push kyrylprojector/experiments:latest

docker build -t kyrylprojector/pipeline:latest --target pipeline .
docker push kyrylprojector/pipeline:latest

docker build -t kyrylprojector/monitoring:latest --target monitoring .
docker push kyrylprojector/monitoring:latest

