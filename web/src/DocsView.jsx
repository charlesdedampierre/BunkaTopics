import { Backdrop, Box, Button, CircularProgress, Container, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";
import React, { useContext, useEffect, useState } from "react";
import { TopicsContext } from "./UploadFileContext";

const bunkaDocs = "bunka_docs.json";
const bunkaTopics = "bunka_topics.json";
const { REACT_APP_API_ENDPOINT } = process.env;

function DocsView() {
  const [docs, setDocs] = useState(null);
  const [topics, setTopics] = useState(null);
  const { data: apiData, isLoading } = useContext(TopicsContext);

  useEffect(() => {
    if (REACT_APP_API_ENDPOINT === "local" || apiData === undefined) {
      // Fetch the JSON data locally
      fetch(`/${bunkaDocs}`)
        .then((response) => response.json())
        .then((localData) => {
          setDocs(localData);
          // Fetch the topics data and merge it with the existing data
          fetch(`/${bunkaTopics}`)
            .then((response) => response.json())
            .then((topicsData) => {
              // Set the topics data with the existing data
              setTopics(topicsData);
            })
            .catch((error) => {
              console.error("Error fetching topics data:", error);
            });
        })
        .catch((error) => {
          console.error("Error fetching JSON data:", error);
        });
    } else {
      // Call the function to create the scatter plot with the data provided by TopicsContext
      setDocs(apiData.docs);
      setTopics(apiData.topics);
    }
  }, [apiData]);

  const docsWithTopics =
    docs && topics
      ? docs.map((doc) => ({
          ...doc,
          topic_name: topics.find((topic) => topic.topic_id === doc.topic_id)?.name || "Unknown",
        }))
      : [];

  const downloadCSV = () => {
    // Create a CSV content string from the data
    const csvContent = `data:text/csv;charset=utf-8,${[
      ["Doc ID", "Topic ID", "Topic Name", "Content"], // CSV header
      ...docsWithTopics.map((doc) => [doc.doc_id, doc.topic_id, doc.topic_name, doc.content]), // CSV data
    ]
      .map((row) => row.map((cell) => `"${cell}"`).join(",")) // Wrap cells in double quotes
      .join("\n")}`; // Join rows with newline

    // Create a Blob containing the CSV data
    const blob = new Blob([csvContent], { type: "text/csv" });

    // Create a download URL for the Blob
    const url = URL.createObjectURL(blob);

    // Create a temporary anchor element to trigger the download
    const a = document.createElement("a");
    a.href = url;
    a.download = "docs.csv"; // Set the filename for the downloaded file
    a.click();

    // Revoke the URL to free up resources
    URL.revokeObjectURL(url);
  };

  return (
    <Container fixed>
      <div className="docs-view">
        <h2>Data</h2>
        {isLoading ? (
          <Backdrop open={isLoading} style={{ zIndex: 9999 }}>
            <CircularProgress color="primary" />
          </Backdrop>
        ) : (
          <div>
            <Button variant="contained" color="primary" onClick={downloadCSV} sx={{ marginBottom: "1em" }}>
              Download CSV
            </Button>
            <Box
              sx={{
                height: "1000px", // Set the height of the table
                overflow: "auto", // Add scroll functionality
              }}
            >
              <TableContainer component={Paper}>
                <Table>
                  <TableHead
                    sx={{
                      backgroundColor: "lightblue", // Set background color
                      position: "sticky", // Make the header sticky
                      top: 0, // Stick to the top
                    }}
                  >
                    <TableRow>
                      <TableCell>Doc ID</TableCell>
                      <TableCell>Topic ID</TableCell>
                      <TableCell>Topic Name</TableCell>
                      <TableCell>Content</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {docsWithTopics.map((doc, index) => (
                      <TableRow
                        key={doc.doc_id}
                        sx={{
                          borderBottom: "1px solid lightblue", // Add light blue border
                        }}
                      >
                        <TableCell>{doc.doc_id}</TableCell>
                        <TableCell>{doc.topic_id}</TableCell>
                        <TableCell>{doc.topic_name}</TableCell>
                        <TableCell>{doc.content}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          </div>
        )}
      </div>
    </Container>
  );
}

export default DocsView;
