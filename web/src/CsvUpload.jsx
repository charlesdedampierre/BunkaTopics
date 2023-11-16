import React, { useState } from "react";
import { Button, Container, Typography } from "@mui/material";

function CsvUpload() {
  const [file, setFile] = useState(null);

  const handleFileUpload = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
  };

  const handleUpload = () => {
    // Implement the logic to upload the file to your API here
    // You may need to make a network request to send the file data
  };

  return (
    <Container>
      <Typography variant="h5" gutterBottom>
        Upload a CSV File
      </Typography>
      <input type="file" accept=".csv" onChange={handleFileUpload} style={{ display: "none" }} id="csv-upload-input" />
      <label htmlFor="csv-upload-input">
        <Button variant="contained" color="primary" component="span">
          Choose File
        </Button>
      </label>
      <Typography variant="subtitle1" gutterBottom>
        {file ? `Selected file: ${file.name}` : "No file selected"}
      </Typography>
      <Button variant="contained" color="primary" disabled={!file} onClick={handleUpload}>
        Upload CSV
      </Button>
    </Container>
  );
}

export default CsvUpload;
