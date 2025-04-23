# Phil's Project Notes

## Table of Contents
- [Project Overview](#project-overview)
- [Model](#model)
- [API](#api)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)

## Project Overview
The main learning from this project was experience in building an API for an AI model using python.
This included tweaking the provided model generation code, developing a method for evaluating the models generated,
Writing an API for interfacing with the model(s) generated, deploying the project to a Docker container, and writing some
test scripts for testing the deployed application.

The project is hosted at https://github.com/pjorgen/aiml-housing-project

## Model
The basic model generation code was provided. 
I decided to tweak it to generate a model with better fit for the given data.
The `model/create_model.py` file is the result of my tweaking, which replaced the regressor with the RandomForestRegressor
and added an iterator to give the code a configurable number of attempts to generate a model. This allows
the randomness inherent in the scikit methods to optimize somewhat better than just taking the first iteration.

I also took the opportunity to add some parameters to the python script so it can be run with a wider variety
of input parameters without requiring code updates. These include adding a model name, a flag for which set of columns
to train on, and a configurable number of iterations to perform to maximize the model. The generation code uses the R2
score as a simplistic evaluation to find the "best" result.

I wrote a separate script to evaluate the model using a set of different metrics. This script also takes a model name as
a parameter to allow it to be run on different models. The evaluation metrics I decided to use are:
- Mean Squared Error (MSE)
- R-squared (R2)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Bias (Average Prediction)

Each metric tells a story, and including a variety of them allows more comprehensive evaluation based on the use case. 
The value of each metric is out of scope for this writeup.

## API
The API is written in a Conda Python environment and makes use of Flask as an application framework. I decided to use
Pydantic as a model layer to automate the validation of requests - each prediction request is deserialized into a 
Pydantic class that ensures the request includes the required information. I ended up using some generic typing to 
reduce code duplication across class definitions, with a base class and two child classes that each expect a certain set
of prediction input columns. The response is also a class, which allows for expedited creation and serialization.

I broke out the prediction code to a separate python file to give the app some better separation of concerns. The 
prediction code is responsible for loading the given model, merging the server-side demographic data, and sending the
merged data to the model. The returned prediction is then packaged into a response class and returned to the API.

The API index dynamically returns a JSON object that lists all of the available endpoints. The `/models` endpoint 
returns a list of the available models, and the `/predict` and `/predict-basic` endpoints are for submitting a request.
There is some error handling in the prediction endpoints for ensuring the request has the required columns by way of 
Pydantic deserialization and validation. There is also API-level exception catching for unexpected issues.

## Docker Deployment
The next step of the project was to package and deploy the app as a docker container, primarily for local deployment and
testing. I ended up writing a few helper scripts for this portion:
- `Dockerfile` was used to define the container. The file is split into multiple phases for efficiency of redeployment
  which really helped during development by cutting the redeploy time down to about 20 seconds from closer to 45,
  allowing faster iteration
- `docker-compose` is a configuration file used to pass environment variables into the Flask app
- `build-docker.ps1` is a shortcut script for building the docker container using standard parameters
- `redeploy.ps1` is a second shortcut script that does a `compose down`, runs the build script, and does a `compose up`

The helper files were really nice for iterating more quickly, especially when in the testing phase and working through 
the bugs I found. Another optimization I made in the docker deployment was to use a micromamba environment instead of a
conda environment. They are compatible, yet with the optimized package list the ensuing docker container was 67% smaller.

## Testing
Each published endpoint in the API is included in the test script, including the `/` index and `/models` endpoints.
The majority of the testing is done on the prediction endpoints, which are each run through every row of the given 
`future_unseen_examples.csv` file. 

## Scaling Up
There are two main ways this application can scale or change over time. New models can be trained and deployed, or usage
can scale up and down over time. My proposed solution to both of these problems are to deploy the application containers
in a kubernetes cluster behind an nginx load balancer. The cluster can be configured to auto-scale up and down based on 
usage, which takes care of that scaling axis. 

By deploying in a cluster, new models can be rolled out by cycling nodes from the older container to the newer one. This
is supported by the `/models` endpoint that returns the available models - the user can request which models are 
available, and request to use one of them. When a new version of the contianer is deployed with updated or new models, 
the model manifest is updated and the user would see the new options. As nodes are cycled out of the cluster and new nodes
take their place, the overall cluster would be updated to support the new models. The details of this deployment would 
need to account for ensuring that users of one node type are segregated to that subcluster to avoid a case where a user 
sees the new v2 model as available, but their prediction request goes to an older node that doesn't support v2 yet.

## Potential Enhancements
There are a few things I'd do to enhance this project that are out of scope at this time. These include:
- Spend more time on model tweaking, using additional evaluation metrics and weighting them to develop a better model
- More error handling of the API endpoints
- Further refinement and optimization of the docker container
- Additional testing of edge cases and error conditions
- Sample kubernetes deployment that runs as a cluster in a local docker deployment behind an nginx load balancer