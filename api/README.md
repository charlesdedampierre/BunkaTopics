# BunkaTopics API built upon FastAPI with Celery and Redis

## Introduction

This README provides an overview of the internals of this API using FastAPI, Celery, and Redis. FastAPI is a modern, fast web framework for building APIs with Python. Celery is a distributed task queue for processing asynchronous tasks. Redis is used as the backend for Celery, storing task states and results.

## Table of Contents

- [BunkaTopics API built upon FastAPI with Celery and Redis](#bunkatopics-api-built-upon-fastapi-with-celery-and-redis)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Server Environment](#setting-up-the-server-environment)
    - [NVIDIA GPU Toolkit Installation Guide for Ubuntu Server](#nvidia-gpu-toolkit-installation-guide-for-ubuntu-server)
      - [Step 1: Check GPU Compatibility](#step-1-check-gpu-compatibility)
      - [Step 2: Update Ubuntu Packages](#step-2-update-ubuntu-packages)
      - [Step 4: Download NVIDIA CUDA Toolkit](#step-4-download-nvidia-cuda-toolkit)
      - [Step 5: Disable Nouveau Drivers (If Necessary)](#step-5-disable-nouveau-drivers-if-necessary)
  - [FastAPI Application Schema](#fastapi-application-schema)
  - [FastAPI, Celery, and Redis Flow](#fastapi-celery-and-redis-flow)
  - [Running the Application](#running-the-application)
  - [Updating the application](#updating-the-application)
  - [Endpoints and Task Management](#endpoints-and-task-management)
  - [Monitoring and Scaling](#monitoring-and-scaling)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker
- Make
- Nginx

## Setting Up the Server Environment

1. Copy .env.model to .env
2. Create the Docker shared network: `make docker_create_network`
3. Install GPU toolkit for python containerized code to use it, *OR* use an VPS Ubuntu Image with NVIDIA cuda already baked in (we advice you to use this last option)

Here's a simple guide to install the NVIDIA GPU Toolkit on an Ubuntu server:

### NVIDIA GPU Toolkit Installation Guide for Ubuntu Server

This guide outlines the steps to install the NVIDIA GPU Toolkit (CUDA) on an Ubuntu server, enabling TensorFlow and other Python libraries to access the GPU.

#### Step 1: Check GPU Compatibility

- Ensure your NVIDIA GPU is compatible with the CUDA version you plan to install.

#### Step 2: Update Ubuntu Packages

- Open a terminal and update your Ubuntu package lists:
  
  ```bash
  sudo apt-get update
  sudo apt-get upgrade
  ```

#### Step 4: Download NVIDIA CUDA Toolkit

- Go to the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads) and select the appropriate version for your Ubuntu server.
- Download the debian (network) installer and follow the instructions on their website.

#### Step 5: Disable Nouveau Drivers (If Necessary)

- Nouveau (default Linux GPU driver) can interfere with NVIDIA drivers. To disable:

  ```bash
  sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
  sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
  sudo update-initramfs -u
  ```

## FastAPI Application Schema

- User: Interacts with the FastAPI application through HTTP requests.
- FastAPI Application: Receives HTTP requests from the user. For asynchronous tasks, it sends task requests to the Celery worker queue and provides immediate responses (like task IDs) back to the user.
 -Celery Worker Queue: Receives task requests from the FastAPI application. It queues these tasks and processes them asynchronously. The status and results of these tasks are managed using Redis.
- Redis: Acts as a message broker and a result backend for Celery. It stores the task queue and the states/results of tasks processed by the Celery workers.

## FastAPI, Celery, and Redis Flow

```plaintext
User Request
  │
  └───> FastAPI Application ──────> Celery Worker Queue ────> Redis
         │                                                           │
         │<──────────────────────────────────────────────────────────┘
         │
         └───> Status/Result Query
```

- User Request: The user sends a request to the FastAPI application (e.g., to process data, start a background job, etc.).

- FastAPI Processes Request: FastAPI processes the request. If it's an asynchronous task, FastAPI sends this task to the Celery worker queue and returns a response (like a task ID) to the user.

- Task Queuing in Redis: The task is queued in Redis by the Celery worker.

- Task Processing: The Celery worker processes the task. The status and result of the task are updated in Redis.

- Result Retrieval: The user can query the FastAPI application for the status or result of the task, which FastAPI retrieves from Redis via the Celery worker.

## Running the Application

Run both API and Celery worker with Docker using the Makefile at the root of this repository

- Start a redis docker : `make docker_run_redis`
- Configure Nginx reverse proxy: `make install_nginx_config` (this uses a Nginx started on your host, not a container)
- Build both docker:  `make build_docker && make build_docker_worker`
- Run both docker: `make run_docker && make run_docker_worker`

Type hostname in your *NIX terminal to print your hostname
Go to http://[$hostname] or <http://localhost>

## Updating the application

- Via git:

```bash
git clone https://github.com/charlesdedampierre/BunkaTopics.git
cd BunkaTopics
git pull # only for an existing clone
```

- Then repeat the process for running the application described above

## Endpoints and Task Management

- The complete API documentation and playground is available at `http://[$hostname]/api/docs`

## Monitoring and Scaling

- Celery worker is in *solo* mode, so the NLP processing tasks run in band.
- Log tasks : `docker logs -f bunkaworker`
- Log API : `docker logs -f bunkaapi`

## Best Practices

- TODO : add user authentication and let them save theirs projects and maps.
- Security considerations : run your server behind a password.

## Troubleshooting

- Look for Docker, FastAPI, Celery, and Redis documentation for common issues and their solutions.
