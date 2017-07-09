# Sentiment Prediction Dashboard README

* * *

## Running the application locally

The Sentiment Prediction Dashboard Application is built using two services, the api-gateway and prediction-server, which are built using Node.js and Python/Fask respectively. To make this application easier to run locally, both services have been dockerized. To run these applications, you can build each container seperately, or run them using docker-compose. A docker-compose.yml file has been provided in the root folder.

The Node.js application interfaces with the Google Natural Language API. To run the application, you must first setup an account on Google Cloud Platform and setup a project that enables the Google Natural Language API.

Ensure that the environment variabel GCP_PROJECT is set to your project's id before running the application.

`export GCP_PROJECT=[your project's id]`

### Building Images and Running Containers Using  Docker-Compose

To run the application using docker-compose ensure that docker and docker-compose are both installed locally. To run these docker-containers on a MAC, you can install docker-machine.

To start up the containers, navigate to the root directory and run:

`docker-compose up`

This will build the images of the api-gateway and prediction-server if they are not present, and will run them once they are built. 

Navigate to http://localhost:3000/, to access the api-gateway. Note: If using docker-machine, you'll have to access PORT 3000 on the the url that docker-machine is exposed on. You can find out this url by running:

`docker-machine ip`

At this url, you can access the api-gateway which will serve the static files of the dashboard on port 3000. You can now use the application.


### Continued Augmentations and Implementations

Implement Service Discovery Using Netflix Eureka and HashiCorp Consul.
