créé une app sur ton heroku lstm-app

cd lstm-api-omar/front
heroku login
heroku container:login
heroku stack:set container -a lstm-app 
docker buildx build --platform linux/amd64 -t lstm-app .
docker tag lstm-app registry.heroku.com/lstm-app/web
docker push registry.heroku.com/lstm-app/web
heroku container:release web -a lstm-app   

app dispo ici:  https://lstm-app-deaa627f967b.herokuapp.com/dashboard