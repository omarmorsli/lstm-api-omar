créé une app sur ton heroku lstm-api

cd lstm-api-omar/back
heroku login
heroku container:login
docker buildx build --platform linux/amd64 -t lstm-api .
docker tag lstm-api registry.heroku.com/lstm-api/web
docker push registry.heroku.com/lstm-api/web
heroku container:release web -a lstm-api   

app dispo ici:  https://lstm-api-ce674ddbb5fc.herokuapp.com/docs