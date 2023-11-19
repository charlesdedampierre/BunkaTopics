# Stage 1: Build the React app
FROM node:20 AS build

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json to install dependencies
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the source code to the container
COPY . .

# docker build --build-arg REACT_APP_API_ENDPOINT=$REACT_APP_API_ENDPOINT -t bunkafront .
ARG REACT_APP_API_ENDPOINT
ENV REACT_APP_API_ENDPOINT $REACT_APP_API_ENDPOINT

# Build the React app
RUN npm run build

# Stage 2: Set up Nginx and serve the React app
FROM nginx:latest

# Copy the built React app from the previous stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy the custom default server block configuration for Nginx
COPY ./nginx-configuration.conf /etc/nginx/conf.d/default.conf

# Expose port 80 for Nginx
EXPOSE 80

# Start Nginx when the container runs
CMD ["nginx", "-g", "daemon off;"]