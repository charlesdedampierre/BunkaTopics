import CloudUploadIcon from "@mui/icons-material/CloudUpload";
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
  Typography,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import Papa from "papaparse";
import React, { useContext, useState } from "react";
import { TopicsContext } from "./UploadFileContext";
import { LABELS } from "./DropdownMenu";

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
  const [selectedView, setSelectedView] = useState("map");
  const [xLeftWord, setXLeftWord] = useState("");
  const [xRightWord, setXRightWord] = useState("");
  const [yTopWord, setYTopWord] = useState("");
  const [yBottomWord, setYBottomWord] = useState("");
  const [radiusSize, setRadiusSize] = useState("");
  const { uploadFile, isLoading } = useContext(TopicsContext);

  const handleClose = () => {
    setOpenSelector(false);
  };

  const handleOpen = () => {
    setOpenSelector(true);
  };

  const handleSelectView = (event) => {
    setSelectedView(`${event.target.value}`);
  };

  const parseCSVFile = (file, sampleSize = 500) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        const csvData = e.target.result;
        const lines = csvData.split("\n");

        // Take a sample of the first 500 lines
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

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);

    if (!file) return;
    // prepare data for the preview Table
    try {
      const parsedData = await parseCSVFile(file);
      setFileData(parsedData);
      setSelectedColumn(""); // Clear the selected column when a new file is uploaded
      handleOpen();
    } catch (exc) {
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

  const handleProcessTopics = async () => {
    if (selectedColumnData.length === 0) return;

    if (selectedFile) {
      uploadFile(selectedFile, {
        nClusters: 10, // TODO set the desired number of clusters in UI ?
        selectedColumn,
        selectedView,
        xLeftWord,
        xRightWord,
        yTopWord,
        yBottomWord,
        radiusSize,
      });
    }
  };

  return (
    <Container component="form">
      <Typography variant="h4" gutterBottom>
        CSV File Viewer
      </Typography>
      <Box marginBottom={2}>
        <Button component="label" variant="outlined" startIcon={<CloudUploadIcon />}>
          Upload a file with at least one column containing text
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
      {isLoading ? (
        <Backdrop open={isLoading} style={{ zIndex: 9999 }}>
          <CircularProgress color="primary" />
        </Backdrop>
      ) : (
        // Content when not loading
        <div>
          {selectedColumnData.length > 0 && (
            <TableContainer component={Paper} style={{ maxHeight: "400px", overflowY: "auto" }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>{selectedColumn}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {selectedColumnData.map((cell, index) => (
                    <TableRow key={`${cell}`}>
                      <TableCell>{cell}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          <Box marginTop={2} display="flex">
            <Button variant="contained" color="primary" onClick={handleProcessTopics} disabled={selectedColumnData.length === 0 || isLoading}>
              {isLoading ? "Processing..." : "Process Topics"}
            </Button>
            <FormControl variant="outlined" className="dropdown-menu" sx={{ minWidth: "200px", marginTop: "1em" }}>
              <InputLabel htmlFor="view-select">Select a View</InputLabel>
              <Select
                label="Select a View"
                value={selectedView}
                onChange={handleSelectView}
                inputProps={{
                  name: "prop-view-select",
                  id: "prop-view-select",
                }}
              >
                <MenuItem value="map">{LABELS.map}</MenuItem>
                <MenuItem value="bourdieu">{LABELS.bourdieu}</MenuItem>
              </Select>
            </FormControl>
            {selectedView === "bourdieu" && (
              <>
                <TextField id="input-bourdieu-xl" label="X left word" variant="outlined" onChange={setXLeftWord} />
                <TextField id="input-bourdieu-xr" label="X right word" variant="outlined" onChange={setXRightWord} />
                <TextField id="input-bourdieu-yt" label="Y top word" variant="outlined" onChange={setYTopWord} />
                <TextField id="input-bourdieu-yb" label="Y bottom word" variant="outlined" onChange={setYBottomWord} />
                <TextField id="input-bourdieu-radius" label="Radius Size" variant="outlined" onChange={setRadiusSize} />
              </>
            )}
          </Box>
        </div>
      )}
    </Container>
  );
}

export default QueryView;
