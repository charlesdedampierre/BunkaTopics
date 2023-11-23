import { Alert, Box, LinearProgress, Typography } from "@mui/material";
import axios from "axios";
import md5 from "crypto-js/md5";
import PropTypes from "prop-types";
import React, { createContext, useCallback, useEffect, useMemo, useState } from "react";

// Create the Context
export const TopicsContext = createContext();

/**
 * Return a MD5 from the contents of a text file.
 * @param {Promise<FileReader>} file
 * @returns String
 */
async function hashPartialFile(file) {
  const reader = new FileReader();
  reader.readAsText(file);

  return new Promise((resolve, reject) => {
    reader.onload = async (event) => {
      const text = event.target.result;
      const lines = text.split("\n");

      let contentToHash;
      if (lines.length < 5) {
        // If less than 5 lines, use the entire content
        contentToHash = text;
      } else {
        // Otherwise, use the first two and the last two lines
        contentToHash = [...lines.slice(0, 2), ...lines.slice(-2)].join("\n");
      }

      // Compute MD5 hash
      const hash = md5(contentToHash).toString();
      resolve(hash);
    };

    reader.onerror = (error) => {
      reject(error);
    };
  });
}

/*
const saveDataToFile = (fileName, data) => {
  const blob = new Blob([data], { type: "application/json" });

  // Create a link element
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = fileName;

  // Trigger a click event to download the file
  a.click();
};
*/
const { REACT_APP_API_ENDPOINT } = process.env;

const ENDPOINT_PATH = "/topics/csv";
// Fetcher function
const fetcher = (url, data) =>
  axios
    .post(url, data, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then((res) => res.data);

// Provider Component
export function TopicsProvider({ children, onSelectView }) {
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState();
  const [error, setError] = useState();
  const [errorText, setErrorText] = useState("");
  const [taskProgress, setTaskProgress] = useState(0); // Add state for task progress
  const [taskID, setTaskID] = useState(null); // Add state for task ID

  const fetchTaskResult = async (taskId) => {
    try {
      const response = await fetch(`/tasks/${taskId}/result`);
      if (response.ok) {
        const data = await response.json();
        setData(data);
      } else {
        console.error("Failed to fetch task result");
      }
    } catch (error) {
      console.error("Error fetching task result:", error);
    }
  };

  const monitorTaskProgress = (taskId) => {
    const evtSource = new EventSource(`/tasks/${taskId}/progress`);
    evtSource.onmessage = function (event) {
      const data = JSON.parse(event.data);
      console.log("Task Progress:", data);
      setTaskProgress(data.progress); // Update progress in state
      if (data.state === "SUCCESS") {
        evtSource.close();
        fetchTaskResult(taskId); // Fetch the task result
        if (onSelectView) onSelectView("map");
      } else if (data.state === "FAILURE") {
        evtSource.close();
      }
    };
  };

  // Handle File Upload and POST Request
  const uploadFile = useCallback(
    async (file, params) => {
      setIsLoading(true);
      setErrorText("");

      try {
        // Generate SHA-256 hash of the file
        const fileHash = await hashPartialFile(file);

        const formData = new FormData();
        formData.append("file", file);
        // Append additional parameters to formData
        formData.append("n_clusters", params.n_clusters);
        formData.append("openapi_key", params.openapi_key);
        formData.append("selected_column", params.selected_column);

        const apiURI = `${REACT_APP_API_ENDPOINT}${ENDPOINT_PATH}?md5=${fileHash}`;
        // Perform the POST request
        const response = await fetcher(apiURI, formData);
        setTaskID(response.task_id);
        monitorTaskProgress(response.task_id); // Start monitoring task progress
      } catch (errorExc) {
        // Handle error
        const errorMessage = errorExc.response?.data?.message || errorExc.message || "An unknown error occurred";
        console.error("Error:", errorMessage);
        setErrorText(errorMessage);
        setError(errorExc.response);
      } finally {
        setIsLoading(false);
      }
    },
    [monitorTaskProgress],
  );

  useEffect(() => {
    if (error?.length) {
      const message = error.response ? error.response.data.message : error.message;
      setErrorText(`Error uploading file.\n${message}`);
      console.error("Error uploading file:", message);
    }
  }, [error]);

  const providerValue = useMemo(
    () => ({
      data,
      uploadFile,
      isLoading,
      error,
    }),
    [data, uploadFile, isLoading, error],
  );

  return (
    <TopicsContext.Provider value={providerValue}>
      <>
        {isLoading && <div className="loader" />}
        {/* Display a progress bar based on task progress */}
        {taskID && (
          <Box display="flex" alignItems="center">
            <Box width="100%" mr={1}>
              <LinearProgress variant="determinate" value={taskProgress} />
            </Box>
            <Box minWidth={35}>
              <Typography variant="body2" color="textSecondary">{`${Math.round(taskProgress)}%`}</Typography>
            </Box>
          </Box>
        )}

        {errorText && (
          <Alert severity="error" className="errorMessage">
            {errorText}
          </Alert>
        )}
        {children}
      </>
    </TopicsContext.Provider>
  );
}

TopicsProvider.propTypes = {
  children: PropTypes.func.isRequired,
  onSelectView: PropTypes.func.isRequired,
};
