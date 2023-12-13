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
  RadioGroup,
  Radio,
  FormControlLabel,
  FormLabel,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import Papa from "papaparse";
import React, { useContext, useState } from "react";
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
  const [xLeftWord, setXLeftWord] = useState("left");
  const [xRightWord, setXRightWord] = useState("right");
  const [yTopWord, setYTopWord] = useState("top");
  const [yBottomWord, setYBottomWord] = useState("bottom");
  const [radiusSize, setRadiusSize] = useState(0.5);
  const [nClusters, setNClusters] = useState(15);
  const [minCountTerms, setMinCountTerms] = useState(1);
  const [nameLength, setNameLength] = useState(3);
  const [cleanTopics, setCleanTopics] = useState(false);
  const [language, setLanguage] = useState("english");
  const { uploadFile, isLoading, selectedView } = useContext(TopicsContext);

  const handleClose = () => {
    setOpenSelector(false);
  };

  const handleOpen = () => {
    setOpenSelector(true);
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

  return (
    <Container component="form">
      <Box marginBottom={2}>
        <Button component="label" variant="outlined" startIcon={<CloudUploadIcon />}>
          Upload a CSV with at least one text column
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
          <Box marginTop={2} display="flex" alignItems="center" flexDirection="column">
            <Button variant="contained" color="primary" onClick={handleProcessTopics} disabled={selectedColumnData.length === 0 || isLoading}>
              {isLoading ? "Processing..." : "Process Topics"}
            </Button>
            {selectedView === "bourdieu" && (
              <FormControl variant="outlined" sx={{ marginTop: "1em", marginLeft: "1em" }}>
                <TextField required id="input-bourdieu-xl" sx={{ marginBottom: "1em" }} label="X left words (comma separated)" variant="outlined" onChange={e => setXLeftWord(e.target.value)} value={xLeftWord} />
                <TextField required id="input-bourdieu-xr" sx={{ marginBottom: "1em" }} label="X right words (comma separated)" variant="outlined" onChange={e => setXRightWord(e.target.value)} value={xRightWord} />
                <TextField required id="input-bourdieu-yt" sx={{ marginBottom: "1em" }} label="Y top words (comma separated)" variant="outlined" onChange={e => setYTopWord(e.target.value)} value={yTopWord} />
                <TextField required id="input-bourdieu-yb" sx={{ marginBottom: "1em" }} label="Y bottom words (comma separated)" variant="outlined" onChange={e => setYBottomWord(e.target.value)} value={yBottomWord} />
                <TextField required id="input-bourdieu-radius" label="Radius Size" variant="outlined" onChange={e => setRadiusSize(e.target.value)} value={radiusSize} />
              </FormControl>
            )}
            <FormControl variant="outlined" sx={{ marginTop: "1em", marginLeft: "1em" }}>
              <TextField required id="input-map-nclusters" sx={{ marginBottom: "1em" }} label="N° Clusters" variant="outlined" onChange={e => setNClusters(e.target.value)} value={nClusters} />
              <TextField required id="input-map-namelength" sx={{ marginBottom: "1em" }} label="Name length" variant="outlined" onChange={e => setNameLength(e.target.value)} value={nameLength} />
              <TextField required id="input-map-mincountterms" sx={{ marginBottom: "1em" }} label="Min Count Terms" variant="outlined" onChange={e => setMinCountTerms(e.target.value)} value={minCountTerms} />
              <RadioGroup required name="cleantopics-radio-group" defaultValue={cleanTopics} onChange={e => setCleanTopics(e.target.value)} variant="outlined" sx={{ marginBottom: "1em" }}>
                <FormLabel id="clean-topics-group-label">Clean Topics</FormLabel>
                <FormControlLabel value={true} label="Yes" control={<Radio />} />
                <FormControlLabel value={false} label="No" control={<Radio />} />
              </RadioGroup>
              <RadioGroup required name="language-radio-group" defaultValue={language} onChange={e => setLanguage(e.target.value)} variant="outlined" sx={{ marginBottom: "1em" }} >
                <FormLabel id="language-group-label">Language</FormLabel>
                <FormControlLabel value="french" label="fr" control={<Radio />} />
                <FormControlLabel value="english" label="en" control={<Radio />} />
              </RadioGroup>
            </FormControl>
          </Box>
        </div>
      )}
    </Container>
  );
}

export default QueryView;
