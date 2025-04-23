# Stage 1: Build conda env
# Start from a base container image of micromamba (it's smaller than miniconda)
FROM mambaorg/micromamba:latest AS builder

# Set the working directory
WORKDIR /app

# Copy the conda config file
COPY conda_environment.yml .

# Run the command to create the environment
RUN micromamba create -y -n housing -f conda_environment.yml

# Stage 2 - final image
FROM mambaorg/micromamba:latest

#Set the working directory
WORKDIR /app

# Choose what to include in the image (everything in app folder)
COPY --from=builder /opt/conda/envs/housing /opt/conda/envs/housing
COPY app/ .
COPY model/data/zipcode_demographics.csv data/

# Set the PATH variable (updated to new key=value format)
ENV PATH=/opt/conda/envs/housing/bin:$PATH

# Set the default command when running the container
CMD ["python", "app.py"]
