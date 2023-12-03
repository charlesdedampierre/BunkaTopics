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
const axiosnInstance = axios.create({
  baseURL: REACT_APP_API_ENDPOINT !== local ? REACT_APP_API_ENDPOINT : undefined
});

const TOPICS_ENDPOINT_PATH = "/topics/csv";
const BOURDIEU_ENDPOINT_PATH = "/bourdieu/csv";

// Fetcher function
const fetcher = (url, data) =>
  axiosnInstance.post(url,
    data,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      }
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


  const monitorTaskProgress = async (selectedView, taskId) => {
    const evtSource = new EventSource(`${REACT_APP_API_ENDPOINT}/tasks/${selectedView === 'map' ? "topics" : "bourdieu"}/${taskId}/progress`);
    evtSource.onmessage = function (event) {
      const data = JSON.parse(event.data);
      console.log("Task Progress:", data);
      const progress = !isNaN(parseInt(data.progress, 10)) ? parseInt(data.progress, 10) : 0;
      setTaskProgress(data.progress); // Update progress in state
      if (data.state === "SUCCESS") {
        const result = JSON.parse(data.result);
        setData(result);
        evtSource.close();
        if (onSelectView) onSelectView("map");
      } else if (data.state === "FAILURE") {
        setError(data.error);
        setTaskProgress(0);
        setIsLoading(false);
        evtSource.close();
      }
    };
  };

  // Handle File Upload and POST Request
  const uploadFile = useCallback(
    async (file, params) => {
      setIsLoading(true);
      setErrorText("");
      const {
        nClusters,
        selectedColumn,
        selectedView,
        xLeftWord,
        xRightWord,
        yTopWord,
        yBottomWord,
        radiusSize
      } = params;

      try {
        // Generate SHA-256 hash of the file
        const fileHash = await hashPartialFile(file);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("selected_column", selectedColumn);
        // Append additional parameters to formData
        if (selectedView === "map") {
          formData.append("n_clusters", nClusters);
        } else if (selectedView === "bourdieu") {
          formData.append("x_left_word", xLeftWord);
          formData.append("x_right_word", xRightWord);
          formData.append("y_top_word", yTopWord);
          formData.append("y_bottom_word", yBottomWord);
          formData.append("radius_size", radiusSize);
        }
        const apiURI = `${selectedView === "map" ? TOPICS_ENDPOINT_PATH : BOURDIEU_ENDPOINT_PATH}?md5=${fileHash}`;
        // Perform the POST request
        const response = await fetcher(apiURI, formData);
        setTaskID(response.task_id);
        await monitorTaskProgress(selectedView, response.task_id); // Start monitoring task progress
      } catch (errorExc) {
        // Handle error
        setError(errorExc);
      } finally {
        setIsLoading(false);
      }
    },
    [monitorTaskProgress],
  );

  /**
   * Handle request errors
   */
  useEffect(() => {
    if (error) {
      const message = error.response?.data?.message || error.message || `${error}` || "An unknown error occurred";
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
