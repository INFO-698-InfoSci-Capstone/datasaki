# FastAPI Application with Docker Compose

## Overview

This project includes user registration and login functionality, SQLAlchemy models with Alembic for migrations, logging configuration, and monitoring with Prometheus and Grafana. The application uses ArcticDB for managing datasets and snapshots, with tenant-based access controls.

## Features

- **User Management**: User registration and login with password hashing.
- **Database Management**: SQLAlchemy ORM with Alembic for migrations and PostgreSQL as the database backend.
- **Logging**: Configured logging for monitoring application activity.
- **Monitoring**: Integration with Prometheus and Grafana for real-time monitoring and visualization.
- **Data Management**: ArcticDB for storing and managing datasets and snapshots with tenant isolation.

## Prerequisites

Ensure you have the following installed on your machine:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Getting Started

### Clone the Repository

git clone <repository-url>
cd <repository-directory>
Build and Run the Application
Build the Docker images and start the services:

docker-compose up --build


Access the application:
FastAPI application: http://localhost:8060
Swagger UI for API documentation: http://localhost:8060/docs


Access Prometheus:
http://localhost:9090


Access Grafana:
http://localhost:3000
Default login: admin / admin

Access Adminer (for database management):
http://localhost:8080


Configuration
The application connects to a PostgreSQL database running in a Docker container.
You can change database credentials and other environment variables in the docker-compose.yml file.


Database Migration
To initialize the database with the latest schema, run the following command inside the web container:

docker-compose exec web alembic upgrade head


Logging
The application uses standard logging. Logs can be found in the console output of the web service.

API Documentation
You can find the API documentation and interactive endpoints using Swagger UI at:

http://localhost:8060/docs


Notes
Ensure that your code changes are reflected in the Docker container by running:
docker-compose up --build


If you want to persist data in your PostgreSQL database, you can configure a volume in the db service section of the docker-compose.yml.

