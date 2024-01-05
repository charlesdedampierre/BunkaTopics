import React, { useState } from "react";
import PropTypes from "prop-types";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Paper from "@mui/material/Paper";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import ListItemIcon from "@mui/material/ListItemIcon";
import DescriptionIcon from '@mui/icons-material/Description';

export const topicsSizeFraction = (topicsCentroids, topicSize) => {
  const totalSize = topicsCentroids.reduce((sum, topic) => sum + topic.size, 0);
  return Math.round((topicSize / totalSize) * 100);
}

function TextContainer({ topicName, topicSizeFraction, content }) {
  const [selectedDocument, setSelectedDocument] = useState(null);

  const handleDocumentClick = (docIndex) => {
    if (selectedDocument === docIndex) {
      setSelectedDocument(null);
    } else {
      setSelectedDocument(docIndex);
    }
  };

  return (
    <div id="topic-box-container">
      <Box className="topic-box">
        <Box
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            background: "rgb(94, 163, 252)",
            padding: "8px",
            color: "white",
            textAlign: "center",
            borderRadius: "20px", // Make it rounder
            margin: "0 auto", // Center horizontally
            width: "80%", // Adjust the width as needed
          }}
        >
          <Typography variant="h4" style={{ marginBottom: "8px" }}>
            {topicName}
          </Typography>
        </Box>
        <Typography
          variant="h5"
          style={{
            marginBottom: "20px",
            marginTop: "20px",
            textAlign: "center",
          }}
        >
          {topicSizeFraction}% of the Territory
        </Typography>
        <Paper elevation={3} style={{ maxHeight: "70vh", overflowY: "auto" }}>
          <List>
            {content.map((doc, index) => (
              <ListItem button key={`textcontainerdoc-${index}`} onClick={() => handleDocumentClick(index)} selected={selectedDocument === index}>
                <ListItemIcon>
                  <DescriptionIcon /> {/* Display a document icon */}
                </ListItemIcon>
                <ListItemText primary={<span style={{ fontSize: "14px" }}>{doc}</span>} />
              </ListItem>
            ))}
          </List>
        </Paper>
      </Box>
    </div>
  );
}

TextContainer.propTypes = {
  topicName: PropTypes.string.isRequired,
  topicSizeFraction: PropTypes.number.isRequired,
  content: PropTypes.array.isRequired,
};

export default TextContainer;
