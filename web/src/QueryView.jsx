import ScheduleSendIcon from '@mui/icons-material/ScheduleSend';
import {
  Backdrop, // Import Backdrop component
  Box,
  Button,
  CircularProgress, // Import CircularProgress component
  Container,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  RadioGroup,
  Radio,
  FormControlLabel,
  FormLabel,
  Alert
} from "@mui/material";
import { styled } from "@mui/material/styles";
import Papa from "papaparse";
import React, { useContext, useState, useCallback } from "react";
import { TopicsContext } from "./UploadFileContext";

const VisuallyHiddenInput = styled("input")({
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  bottom: 0,
  left: 0,
  whiteSpace: "nowrap",
  width: 1,
});

function QueryView() {
  const [fileData, setFileData] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedColumnData, setSelectedColumnData] = useState([]);
  const [openSelector, setOpenSelector] = React.useState(false);
  const [xLeftWord, setXLeftWord] = useState("past");
  const [xRightWord, setXRightWord] = useState("future");
  const [yTopWord, setYTopWord] = useState("positive");
  const [yBottomWord, setYBottomWord] = useState("negative");
  const [radiusSize, setRadiusSize] = useState(0.5);
  const [nClusters, setNClusters] = useState(15);
  const [minCountTerms, setMinCountTerms] = useState(1);
  const [nameLength, setNameLength] = useState(3);
  const [cleanTopics, setCleanTopics] = useState(false);
  const [language, setLanguage] = useState("english");
  const { uploadFile, isLoading, selectedView, refreshBourdieuQuery } = useContext(TopicsContext);
  const [fileDataTooLong, setFileDataTooLong] = useState(false);
  const [fileDataError, setFileDataError] = useState(null);

  /**
   * Column name selector handler
   */
  const handleClose = () => {
    setOpenSelector(false);
  };

  const handleOpen = () => {
    setOpenSelector(true);
  };

  /**
   * Parse the CSV and take a sample to display the preview
   * @param {*} file 
   * @param {*} sampleSize 
   * @returns 
   */
  const parseCSVFile = (file, sampleSize = 100) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        const csvData = e.target.result;
        const lines = csvData.split("\n");

        setFileDataTooLong(lines.length > 10000);
        // Take a sample of the first 500 lines to display preview
        const sampleLines = lines.slice(0, sampleSize).join("\n");

        Papa.parse(sampleLines, {
          complete: (result) => {
            resolve(result.data);
          },
          error: (parseError) => {
            reject(parseError.message);
          },
        });
      };
      reader.readAsText(file);
    });

  /**
   * Handler the file selection ui workflow
   * @param {Event} e 
   * @returns 
   */
  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);

    if (!file) return;
    // prepare data for the preview Table
    try {
      const parsedData = await parseCSVFile(file);
      setFileData(parsedData);
      setSelectedColumn(""); // Clear the selected column when a new file is uploaded
      if (fileDataTooLong === false) {
        handleOpen();
      }
      else {
        handleClose();
      }
    } catch (exc) {
      setFileDataError("Error parsing the CSV file, please check your file before uploading");
      console.error("Error parsing CSV:", exc);
    }
  };

  const handleColumnSelect = (e) => {
    const columnName = e.target.value;
    setSelectedColumn(columnName);

    // Extract the content of the selected column
    const columnIndex = fileData[0].indexOf(columnName);
    const columnData = fileData.slice(1).map((row) => row[columnIndex]);

    setSelectedColumnData(columnData);
  };

  /**
   * Launch the upload and processing
   */
  const handleProcessTopics = async () => {
    // Return if no column selected
    if (selectedColumnData.length === 0) return;

    if (selectedFile && !isLoading) {
      uploadFile(selectedFile, {
        nClusters,
        selectedColumn,
        selectedView,
        xLeftWord,
        xRightWord,
        yTopWord,
        yBottomWord,
        radiusSize,
        nameLength,
        minCountTerms,
        language,
        cleanTopics
      });
    }
  };

  const handleRefreshQuery = useCallback(async () => {
    if (!isLoading) {
      await refreshBourdieuQuery({
        topic_param: {
          n_clusters: nClusters,
          name_lenght: nameLength,
          min_count_terms: minCountTerms,
          language: language,
          clean_topics: cleanTopics
        },
        bourdieu_query: {
          x_left_words: xLeftWord.split(","),
          x_right_words: xRightWord.split(","),
          y_top_words: yTopWord.split(","),
          y_bottom_words: yBottomWord.split(","),
          radius_size: radiusSize,
        }
      });
    }
  });

  const openTableContainer = selectedColumnData.length > 0 && fileData.length > 0 && fileData.length <= 10000 && fileDataTooLong === false && fileDataError == null;

  return (
    <Container component="form">
      {selectedView === "map" && (
        <>
          <Box marginBottom={2}>
            <Button component="label" variant="outlined" endIcon={<ScheduleSendIcon />}>
              Upload a CSV (max 10 000 lines) and queue processing
              <VisuallyHiddenInput type="file" onChange={handleFileChange} required />
            </Button>
          </Box>
          <Box marginBottom={2}>
            <FormControl variant="outlined" fullWidth>
              <InputLabel>Select a Column</InputLabel>
              <Select value={selectedColumn} onChange={handleColumnSelect} onClose={handleClose} onOpen={handleOpen} open={openSelector}>
                {fileData[0]?.map((header, index) => (
                  <MenuItem key={`${header}`} value={header}>
                    {header}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </>
      )}
      {isLoading ? (
        <Backdrop open={isLoading} style={{ zIndex: 9999 }}>
          <CircularProgress color="primary" />
        </Backdrop>
      ) : (
        // Content when not loading
        <div>
          {openTableContainer && (
            <TableContainer component={Paper} style={{ maxHeight: "400px", overflowY: "auto" }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>{selectedColumn}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {selectedColumnData.map((cell, index) => (
                    <TableRow key={`table-${index}`}>
                      <TableCell>{cell}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          {fileDataTooLong && (
            <Alert severity="error">CSV must have less than 10 000 lines (this is a demo)</Alert>
          )}
          {fileDataError && (
            <Alert severity="error">CSV must have less than 10 000 lines (this is a demo)</Alert>
          )}
          {selectedView === "bourdieu" && (
            <Box marginTop={2} display="flex" alignItems="center" flexDirection="column">
              <FormControl variant="outlined">
                <TextField required id="input-bourdieu-xl" sx={{ marginBottom: "0.5em" }} label="X left words (comma separated)" variant="outlined" onChange={e => setXLeftWord(e.target.value)} value={xLeftWord} />
                <TextField required id="input-bourdieu-xr" sx={{ marginBottom: "1em" }} label="X right words (comma separated)" variant="outlined" onChange={e => setXRightWord(e.target.value)} value={xRightWord} />
                <TextField required id="input-bourdieu-yt" sx={{ marginBottom: "1em" }} label="Y top words (comma separated)" variant="outlined" onChange={e => setYTopWord(e.target.value)} value={yTopWord} />
                <TextField required id="input-bourdieu-yb" sx={{ marginBottom: "1em" }} label="Y bottom words (comma separated)" variant="outlined" onChange={e => setYBottomWord(e.target.value)} value={yBottomWord} />
                <TextField required id="input-bourdieu-radius" sx={{ marginBottom: "1em" }} label="Radius Size" variant="outlined" onChange={e => setRadiusSize(e.target.value)} value={radiusSize} />
                <TextField required id="input-map-nclusters" sx={{ marginBottom: "1em" }} label="N° Clusters" variant="outlined" onChange={e => setNClusters(e.target.value)} value={nClusters} />
                <TextField required id="input-map-namelength" sx={{ marginBottom: "1em" }} label="Name length" variant="outlined" onChange={e => setNameLength(e.target.value)} value={nameLength} />
                <TextField required id="input-map-mincountterms" sx={{ marginBottom: "1em" }} label="Min Count Terms" variant="outlined" onChange={e => setMinCountTerms(e.target.value)} value={minCountTerms} />
                <RadioGroup required name="cleantopics-radio-group" defaultValue={cleanTopics} onChange={e => setCleanTopics(e.target.value)} variant="outlined" sx={{ marginBottom: "1em" }} disabled>
                  <FormLabel id="clean-topics-group-label">Clean Topics</FormLabel>
                  <FormControlLabel value={true} label="Yes" control={<Radio />} disabled />
                  <FormControlLabel value={false} label="No" control={<Radio />} disabled />
                </RadioGroup>
              </FormControl>
              <Button variant="contained" color="primary" onClick={handleRefreshQuery} disabled={isLoading || fileDataTooLong === true || fileDataError !== null}>
                {isLoading ? "Processing..." : "Refresh Bourdieu Axes"}
              </Button>
            </Box>
          )}
          {selectedView === "map" && (
            <Box marginTop={2} display="flex" alignItems="center" flexDirection="column">
              <Button variant="contained" color="primary" onClick={handleProcessTopics} disabled={selectedColumnData.length === 0 || isLoading || fileDataTooLong === true || fileDataError !== null}>
                {isLoading ? "Processing..." : "Process Topics"}
              </Button>
              <FormControl variant="outlined" sx={{ marginTop: "1em", marginLeft: "1em" }}>
                <TextField required id="input-map-nclusters" sx={{ marginBottom: "1em" }} label="N° Clusters" variant="outlined" onChange={e => setNClusters(e.target.value)} value={nClusters} />
                <TextField required id="input-map-namelength" sx={{ marginBottom: "1em" }} label="Name length" variant="outlined" onChange={e => setNameLength(e.target.value)} value={nameLength} />
                <TextField required id="input-map-mincountterms" sx={{ marginBottom: "1em" }} label="Min Count Terms" variant="outlined" onChange={e => setMinCountTerms(e.target.value)} value={minCountTerms} />
                <RadioGroup required name="cleantopics-radio-group" defaultValue={cleanTopics} onChange={e => setCleanTopics(e.target.value)} variant="outlined" sx={{ marginBottom: "1em" }} disabled>
                  <FormLabel id="clean-topics-group-label">Clean Topics</FormLabel>
                  <FormControlLabel value={true} label="Yes" control={<Radio />} disabled />
                  <FormControlLabel value={false} label="No" control={<Radio />} disabled />
                </RadioGroup>
                <RadioGroup required name="language-radio-group" defaultValue={language} onChange={e => setLanguage(e.target.value)} variant="outlined" sx={{ marginBottom: "1em" }}>
                  <FormLabel id="language-group-label">Language</FormLabel>
                  <FormControlLabel value="french" label="fr" control={<Radio />} />
                  <FormControlLabel value="english" label="en" control={<Radio />} />
                </RadioGroup>
              </FormControl>
            </Box>
          )}
        </div>
      )}
    </Container>
  );
}

export default QueryView;
