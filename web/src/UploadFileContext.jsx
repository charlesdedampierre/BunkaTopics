import { Alert, Box, Typography, Backdrop } from "@mui/material";
import CircularProgress from '@mui/material/CircularProgress';
import axios from "axios";
import PropTypes from "prop-types";
import React, { createContext, useCallback, useEffect, useMemo, useState } from "react";

// Create the Context
export const TopicsContext = createContext();

const { REACT_APP_API_ENDPOINT } = process.env;

const TOPICS_ENDPOINT_PATH = `${REACT_APP_API_ENDPOINT}/topics/csv/`;
const BOURDIEU_ENDPOINT_PATH = `${REACT_APP_API_ENDPOINT}/bourdieu/csv/`;
const REFRESH_BOURDIEU_ENDPOINT_PATH = `${REACT_APP_API_ENDPOINT}/bourdieu/refresh/`;

// Fetcher functions
const postForm = (url, data) =>
  axios
    .post(url, data, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then((res) => res.data);

const postJson = (url, data) =>
  axios
    .post(url, data, {
      headers: {
        "Content-Type": "application/json",
      },
    })
    .then((res) => res.data);

// Provider Component
export function TopicsProvider({ children, onSelectView, selectedView }) {
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState();
  const [bourdieuData, setBourdieuData] = useState();
  const [error, setError] = useState();
  const [errorText, setErrorText] = useState("");
  const [taskProgress, setTaskProgress] = useState(0); // TODO Add state for task progress when the backend is ready
  const [taskID, setTaskID] = useState(null); // Add state for task ID
  const [currentDatasetId, setCurrentDatasetId] = useState(null); // Current Dataset Id equals Task Id for the moment

  const monitorTaskProgress = async (selectedView, taskId) => {
    const evtSource = new EventSource(`${REACT_APP_API_ENDPOINT}/tasks/${selectedView === "map" ? "topics" : "bourdieu"}/${taskId}/progress`);
    evtSource.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        const progress = !isNaN(Math.ceil(data.progress)) ? Math.ceil(data.progress) : 0;
        console.log("Task Progress:", progress);
        setTaskProgress(progress); // Update progress in state
        if (data.state === "SUCCESS") {
          if (selectedView === "map") {
            setData({
              docs: data.result.docs,
              topics: data.result.topics
            });
            setBourdieuData(data.result.bourdieu_response);
          } else if (selectedView === "bourdieu") {
            setBourdieuData(data.result);
          }
          setTaskProgress(100);
          evtSource.close();
          setIsLoading(false);
          setTaskID(null);
          if (onSelectView) onSelectView(selectedView);
        } else if (data.state === "FAILURE") {
          setError(data.error);
          setTaskProgress(0);
          evtSource.close();
          setIsLoading(false);
          evtSource.close();
        }
      } catch (error) {
        console.error("EventSource exception");
        console.error(error);
        setError(error);
        evtSource.close();
        setIsLoading(false);
      }
    };
  };

  // Handle File Upload and POST Request
  const uploadFile = useCallback(
    async (file, params) => {
      setIsLoading(true);
      setErrorText("");
      const { nClusters, selectedColumn, selectedView, xLeftWord, xRightWord, yTopWord, yBottomWord, radiusSize } = params;
      const { nameLength, language, cleanTopics, minCountTerms } = params;

      try {
        // Generate SHA-256 hash of the file
        const formData = new FormData();
        formData.append("file", file);
        formData.append("selected_column", selectedColumn);
        formData.append("n_clusters", nClusters);
        formData.append("name_length", nameLength);
        formData.append("language", language);
        formData.append("clean_topics", cleanTopics);
        formData.append("min_count_terms", minCountTerms);
        // Append bourdieu parameters, processing activated by defaut
        formData.append("process_bourdieu", true);
        formData.append("x_left_words", xLeftWord);
        formData.append("x_right_words", xRightWord);
        formData.append("y_top_words", yTopWord);
        formData.append("y_bottom_words", yBottomWord);
        formData.append("radius_size", radiusSize);

        const apiURI = `${selectedView === "map" ? TOPICS_ENDPOINT_PATH : BOURDIEU_ENDPOINT_PATH}`;
        // Perform the POST request
        const response = await postForm(apiURI, formData);
        setTaskID(response.task_id);
        setCurrentDatasetId(response.task_id);
        await monitorTaskProgress(selectedView, response.task_id); // Start monitoring task progress
      } catch (errorExc) {
        // Handle error
        setError(errorExc);
        setTaskID(null);
        setCurrentDatasetId(null);
      } finally {
        setIsLoading(false);
      }
    },
    [monitorTaskProgress],
  );

  const refreshBourdieuQuery = useCallback(
    async (params) => {
      setIsLoading(true);
      setErrorText("");
      if (currentDatasetId !== null) {
        try {
          const apiURI = `${REFRESH_BOURDIEU_ENDPOINT_PATH}${currentDatasetId}`;
          // Perform the POST request
          const response = await postJson(apiURI, params);
          setBourdieuData(response);
        } catch (errorExc) {
          // Handle error
          setError(errorExc);
        } finally {
          setIsLoading(false);
        }
      } else {
        setIsLoading(false);
        setError("Please import a CSV from the Map view before querying");
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

  /**
   * Shared functions and variables of this TopicsContext and TopicsProvider
   */
  const providerValue = useMemo(
    () => ({
      data,
      bourdieuData,
      uploadFile,
      isLoading,
      error,
      selectedView,
      refreshBourdieuQuery
    }),
    [data, uploadFile, isLoading, error, selectedView, refreshBourdieuQuery],
  );

  // const normalisePercentage = (value) => Math.ceil((value * 100) / 100);

  return (
    <TopicsContext.Provider value={providerValue}>
      <>
        {isLoading && <div className="loader" />}
        {/* Display a progress bar based on task progress */}
        {taskID && (
          <Backdrop
            sx={{ zIndex: 99999 }}
            open={taskID !== undefined}
          >
            <Box display={"flex"} width="30%" alignItems={"center"} flexDirection={"column"} sx={{ backgrounColor: "#FFF", fontSize: 20, fontWeight: 'medium' }}>
              <Box minWidth={200}>
                <Typography variant="h4">Bunka is cooking your data, please wait few seconds</Typography>
              </Box>
              <CircularProgress />
              {/* <Box minWidth={35}>
                <Typography variant="subtitle">{`${normalisePercentage(taskProgress)}%`}</Typography>
              </Box> */}
            </Box>
          </Backdrop>
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
